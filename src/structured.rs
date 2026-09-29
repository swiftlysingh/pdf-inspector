//! Source-preserving structured extraction evidence and coordinate conversion.
//!
//! These types deliberately sit below application-specific interpretation.
//! They describe what the PDF extractor observed, including page-local tables
//! and text lines, without deciding what a value means.

use std::collections::{BTreeMap, HashMap, HashSet};

use lopdf::{Document, Object, ObjectId};
use serde::Serialize;

use crate::extractor::visible_page_box;
use crate::structure_tree::StructRole;
use crate::tables::{Table, TableKind, TableSpan};
use crate::types::{TextItem, TextLine};
use crate::PdfError;

/// Structured output produced by an opt-in [`crate::PdfOptions`] capture.
///
/// Page indexes are zero-based here so transport callers do not have to infer
/// a second numbering convention from the engine's internal one-based pages.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredDocument {
    pub pages: Vec<StructuredPage>,
}

/// One requested source page in structured output.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredPage {
    pub page_index: u32,
    pub size: StructuredPageSize,
    pub status: StructuredPageStatus,
    pub blocks: Vec<StructuredBlock>,
}

/// Upright visible page size in PDF points.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StructuredPageSize {
    pub width: f32,
    pub height: f32,
}

/// Extraction state for a requested page.
///
/// `Extracted` says processing completed, not that the text is guaranteed to
/// match what a person sees in the PDF. `NeedsOcr` may still contain useful
/// partial embedded text. `ExtractionFailed` is reserved for a recoverable
/// per-page failure; fatal document failures continue to return `PdfError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredPageStatus {
    Extracted,
    NeedsOcr,
    ExtractionFailed,
}

/// A block in inferred page reading order.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredBlock {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<StructuredBounds>,
    pub content: StructuredBlockContent,
}

/// Tagged structured block content.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuredBlockContent {
    Text {
        lines: Vec<StructuredTextLine>,
    },
    Table {
        row_count: u32,
        column_count: u32,
        cells: Vec<StructuredTableCell>,
        continuation_rows: Vec<StructuredTableContinuation>,
        unassigned_lines: Vec<StructuredTextLine>,
    },
}

/// A physical detector row joined to an earlier row by shared layout cleanup.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StructuredTableContinuation {
    pub row: u32,
    pub continuation_of: u32,
}

/// One extracted line before Markdown formatting or escaping.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredTextLine {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<StructuredBounds>,
}

/// One anchored table cell. Slots covered by a span are omitted rather than
/// represented as invented blank cells.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredTableCell {
    pub row: u32,
    pub column: u32,
    pub row_span: u32,
    pub column_span: u32,
    pub state: StructuredCellState,
    pub role: StructuredCellRole,
    pub lines: Vec<StructuredTextLine>,
    /// The raw detector-grid value when it cannot be reconstructed from the
    /// positioned line evidence. This preserves uncertainty without replacing
    /// source lines or inventing a location for the detector's normalization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<StructuredBounds>,
}

/// Whether a table slot has extracted source content, is a confirmed blank in
/// a detected grid, or could not be associated with the raw grid confidently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredCellState {
    Extracted,
    Empty,
    Unresolved,
}

/// A role supplied by tagged-PDF evidence, if present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredCellRole {
    Header,
    Body,
    Unknown,
}

/// A normalized visible-page rectangle.
///
/// Coordinates have their origin at the top-left of the page's upright visible
/// CropBox (or MediaBox when no CropBox exists), and are normalized to 0...1.
/// Geometry is omitted instead of fabricated when the source evidence is not
/// finite or does not intersect the visible page.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct StructuredBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Internal visible-page geometry. Source coordinates use the PDF's
/// unrotated, bottom-left-origin user space.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageGeometry {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    rotation: u16,
    size: StructuredPageSize,
    /// Content-stream rotation heuristics change item coordinates without a
    /// corresponding page-dictionary transform that this geometry layer can
    /// prove. Preserve the text but omit locations for those pages.
    bounds_available: bool,
}

impl PageGeometry {
    pub(crate) fn disable_bounds(&mut self) {
        self.bounds_available = false;
    }

    fn source_width(self) -> f32 {
        self.x1 - self.x0
    }

    fn source_height(self) -> f32 {
        self.y1 - self.y0
    }

    fn upright_source_width(self) -> f32 {
        if matches!(self.rotation, 90 | 270) {
            self.source_height()
        } else {
            self.source_width()
        }
    }

    fn upright_source_height(self) -> f32 {
        if matches!(self.rotation, 90 | 270) {
            self.source_width()
        } else {
            self.source_height()
        }
    }
}

#[cfg(test)]
pub(crate) fn test_page_geometry() -> PageGeometry {
    PageGeometry {
        x0: 0.0,
        y0: 0.0,
        x1: 612.0,
        y1: 792.0,
        rotation: 0,
        size: StructuredPageSize {
            width: 612.0,
            height: 792.0,
        },
        bounds_available: true,
    }
}

/// Resolve visible geometry for the selected, one-based engine pages.
pub(crate) fn page_geometries(
    document: &Document,
    pages: &[u32],
) -> Result<BTreeMap<u32, PageGeometry>, PdfError> {
    let page_ids = document.get_pages();
    let mut result = BTreeMap::new();
    for &page in pages {
        let page_id = page_ids
            .get(&page)
            .copied()
            .ok_or(PdfError::InvalidStructure)?;
        result.insert(page, page_geometry(document, page_id)?);
    }
    Ok(result)
}

/// Build pages for a result that did not run text/table extraction, such as a
/// scanned document. Every selected source page remains represented.
pub(crate) fn empty_document(
    geometries: &BTreeMap<u32, PageGeometry>,
    status_by_page: &HashMap<u32, StructuredPageStatus>,
) -> StructuredDocument {
    StructuredDocument {
        pages: geometries
            .iter()
            .map(|(&page, geometry)| StructuredPage {
                page_index: page.saturating_sub(1),
                size: geometry.size,
                status: status_by_page
                    .get(&page)
                    .copied()
                    .unwrap_or(StructuredPageStatus::Extracted),
                blocks: Vec::new(),
            })
            .collect(),
    }
}

/// Add explicit page coverage and final OCR statuses to blocks captured in the
/// shared Markdown/table pipeline.
pub(crate) fn document_from_blocks(
    geometries: &BTreeMap<u32, PageGeometry>,
    mut blocks_by_page: BTreeMap<u32, Vec<StructuredBlock>>,
    status_by_page: &HashMap<u32, StructuredPageStatus>,
) -> StructuredDocument {
    let pages = geometries
        .iter()
        .map(|(&page, geometry)| StructuredPage {
            page_index: page.saturating_sub(1),
            size: geometry.size,
            status: status_by_page
                .get(&page)
                .copied()
                .unwrap_or(StructuredPageStatus::Extracted),
            blocks: blocks_by_page.remove(&page).unwrap_or_default(),
        })
        .collect();
    StructuredDocument { pages }
}

/// Turn source text lines into one structured text block.
pub(crate) fn text_block(lines: &[TextLine], geometry: PageGeometry) -> StructuredBlock {
    let lines: Vec<StructuredTextLine> = lines
        .iter()
        .map(|line| StructuredTextLine {
            text: line.text(),
            bounds: bounds_for_items(&line.items, geometry),
        })
        .collect();
    let bounds = union_bounds(lines.iter().filter_map(|line| line.bounds));
    StructuredBlock {
        id: String::new(),
        bounds,
        content: StructuredBlockContent::Text { lines },
    }
}

/// Preserve detector-claimed source items as ordinary text when a malformed
/// table hypothesis has no logical raw grid. This avoids silently dropping
/// source text merely because Markdown can render a degenerate table.
pub(crate) fn text_block_from_items(items: &[TextItem], geometry: PageGeometry) -> StructuredBlock {
    let lines = source_lines(items, geometry);
    let bounds = union_bounds(lines.iter().filter_map(|line| line.bounds));
    StructuredBlock {
        id: String::new(),
        bounds,
        content: StructuredBlockContent::Text { lines },
    }
}

/// Turn a raw, pre-Markdown table into a structured table block.
///
/// The normal detector does not expose table-structure-recognition cells. We
/// retain only evidence it actually has: the raw grid, tagged TH/TD roles, and
/// vertical spans established from rectangle evidence by
/// `propagate_merged_cells`.
pub(crate) fn table_block(
    table: &Table,
    source_items: &[TextItem],
    geometry: PageGeometry,
    struct_roles: Option<&HashMap<u32, HashMap<i64, StructRole>>>,
) -> StructuredBlock {
    // The raw cell matrix is the detector's only unambiguous logical grid.
    // Some strategies retain row/column edges (N + 1) while others retain
    // centers (N), so using their lengths would manufacture a blank row or
    // column in public output.
    let row_count = table.cells.len();
    let column_count = table.cells.iter().map(Vec::len).max().unwrap_or(0);
    let (mut cell_items, association_reliable) =
        assign_table_items(table, source_items, row_count, column_count, geometry);
    let unassigned_lines = if association_reliable {
        Vec::new()
    } else {
        let claimed_items: Vec<TextItem> = table
            .item_indices
            .iter()
            .filter_map(|&index| source_items.get(index).cloned())
            .collect();
        cell_items = vec![vec![Vec::new(); column_count]; row_count];
        source_lines(&claimed_items, geometry)
    };
    let spans = normalized_spans(&table.spans, row_count, column_count);
    let mut covered = HashSet::new();
    let mut span_at_anchor = HashMap::new();
    for span in spans {
        for row in span.row..span.row + span.row_span {
            for column in span.column..span.column + span.column_span {
                if row != span.row || column != span.column {
                    covered.insert((row, column));
                }
            }
        }
        span_at_anchor.insert((span.row, span.column), span);
    }

    let mut cells = Vec::with_capacity(row_count.saturating_mul(column_count));
    for row in 0..row_count {
        for column in 0..column_count {
            if covered.contains(&(row, column)) {
                continue;
            }
            let span = span_at_anchor
                .get(&(row, column))
                .copied()
                .unwrap_or(TableSpan::single(row, column));
            let mut items = std::mem::take(&mut cell_items[row][column]);
            if span.row_span > 1 || span.column_span > 1 {
                for (span_row, span_cells) in cell_items
                    .iter_mut()
                    .enumerate()
                    .skip(span.row)
                    .take(span.row_span)
                {
                    for (span_column, span_items) in span_cells
                        .iter_mut()
                        .enumerate()
                        .skip(span.column)
                        .take(span.column_span)
                    {
                        if span_row != row || span_column != column {
                            items.append(span_items);
                        }
                    }
                }
            }
            let source_lines = source_lines(&items, geometry);
            let raw_cell = table.cells.get(row).and_then(|cells| cells.get(column));
            let raw_text = raw_cell.map_or("", String::as_str);
            // `lines` always remain positioned source evidence. The raw grid is
            // retained separately only where its text cannot be reconstructed
            // from those lines, rather than replacing the evidence with an
            // invented one-line location.
            let matched_lines = lines_matching_raw_cell(&source_lines, raw_text);
            let state = if raw_cell.is_none() {
                // A ragged detector matrix has no established grid evidence
                // for this public slot. It must not become a confirmed blank.
                StructuredCellState::Unresolved
            } else if raw_text.is_empty() {
                if association_reliable && source_lines.is_empty() {
                    StructuredCellState::Empty
                } else {
                    StructuredCellState::Unresolved
                }
            } else if association_reliable && matched_lines {
                StructuredCellState::Extracted
            } else {
                StructuredCellState::Unresolved
            };
            let detected_text = (state == StructuredCellState::Unresolved
                && !raw_text.is_empty()
                && !matched_lines)
                .then(|| raw_text.to_owned());
            let lines = source_lines;
            let bounds = union_bounds(lines.iter().filter_map(|line| line.bounds));
            let role = cell_role(&items, struct_roles);
            cells.push(StructuredTableCell {
                row: row as u32,
                column: column as u32,
                row_span: span.row_span as u32,
                column_span: span.column_span as u32,
                state,
                role,
                lines,
                detected_text,
                bounds,
            });
        }
    }

    let bounds = union_bounds(
        cells
            .iter()
            .filter_map(|cell| cell.bounds)
            .chain(unassigned_lines.iter().filter_map(|line| line.bounds)),
    );
    let continuation_rows = if table.kind == TableKind::Toc {
        Vec::new()
    } else {
        crate::tables::format::table_layout(&table.cells)
            .continuation_rows
            .into_iter()
            .map(|(row, continuation_of)| StructuredTableContinuation {
                row: row as u32,
                continuation_of: continuation_of as u32,
            })
            .collect()
    };
    StructuredBlock {
        id: String::new(),
        bounds,
        content: StructuredBlockContent::Table {
            row_count: row_count as u32,
            column_count: column_count as u32,
            cells,
            continuation_rows,
            unassigned_lines,
        },
    }
}

/// Assign deterministic document-scoped block IDs after reading-order capture.
pub(crate) fn assign_block_ids(page: u32, blocks: &mut [StructuredBlock]) {
    for (index, block) in blocks.iter_mut().enumerate() {
        block.id = format!("p{}-b{}", page.saturating_sub(1), index);
    }
}

fn source_lines(items: &[TextItem], geometry: PageGeometry) -> Vec<StructuredTextLine> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut items = items.to_vec();
    items.sort_by(|left, right| {
        right
            .y
            .total_cmp(&left.y)
            .then_with(|| left.x.total_cmp(&right.x))
    });
    let tolerance = items
        .iter()
        .map(|item| item.height.abs().max(item.font_size.abs()))
        .filter(|height| height.is_finite() && *height > 0.0)
        .reduce(f32::min)
        .map(|height| (height * 0.45).clamp(1.5, 5.0))
        .unwrap_or(3.0);
    let mut grouped: Vec<Vec<TextItem>> = Vec::new();
    for item in items {
        if let Some(line) = grouped.last_mut().filter(|line| {
            line.first()
                .is_some_and(|first| (first.y - item.y).abs() <= tolerance)
        }) {
            line.push(item);
        } else {
            grouped.push(vec![item]);
        }
    }
    grouped
        .into_iter()
        .map(|mut items| {
            let line_rtl = crate::text_utils::is_rtl_text(items.iter().map(|item| &item.text));
            crate::text_utils::sort_line_items(&mut items, line_rtl);
            let line = TextLine {
                y: items.first().map(|item| item.y).unwrap_or_default(),
                page: items.first().map(|item| item.page).unwrap_or_default(),
                items,
                adaptive_threshold: 0.10,
            };
            StructuredTextLine {
                text: line.text(),
                bounds: bounds_for_items(&line.items, geometry),
            }
        })
        .collect()
}

fn lines_matching_raw_cell(lines: &[StructuredTextLine], raw_text: &str) -> bool {
    if raw_text.is_empty() {
        return lines.is_empty();
    }
    lines
        .iter()
        .flat_map(|line| line.text.split_whitespace())
        .eq(raw_text.split_whitespace())
}

fn assign_table_items(
    table: &Table,
    source_items: &[TextItem],
    row_count: usize,
    column_count: usize,
    geometry: PageGeometry,
) -> (Vec<Vec<Vec<TextItem>>>, bool) {
    if row_count == 0 || column_count == 0 {
        return (vec![vec![Vec::new(); column_count]; row_count], false);
    }
    // The ruled-line detector is identifiable from its N + 1 column edges.
    // It alone retains N leading row edges for N logical rows; other
    // detectors retain anchors or centers, for which a shifted row band is
    // only a hypothesis.
    let rows_support_leading_edges = column_count
        .checked_add(1)
        .is_some_and(|edge_count| table.columns.len() == edge_count);
    let Some(row_axes) = table_axis_candidates(&table.rows, row_count, rows_support_leading_edges)
    else {
        return (vec![vec![Vec::new(); column_count]; row_count], false);
    };
    let Some(column_axes) = table_axis_candidates(&table.columns, column_count, false) else {
        return (vec![vec![Vec::new(); column_count]; row_count], false);
    };
    if table.item_indices.is_empty()
        && table
            .cells
            .iter()
            .flatten()
            .any(|cell| !cell.trim().is_empty())
    {
        return (vec![vec![Vec::new(); column_count]; row_count], false);
    }

    // N detector coordinates can be either source-item anchors or the leading
    // edges of N physical grid bands. Prefer anchors, then use the raw cell
    // strings only to disambiguate an equally valid geometry convention. The
    // raw grid has no positioned item identities, so it cannot replace this
    // association or reconstruct duplicate source occurrences on its own.
    let mut best = assign_items_to_axes(
        table,
        source_items,
        row_count,
        column_count,
        row_axes[0],
        column_axes[0],
    );
    let mut best_score = raw_cell_match_score(&best.0, &table.cells, geometry);
    let mut selected_axes = (row_axes[0], column_axes[0]);
    for (row_index, rows) in row_axes.iter().copied().enumerate() {
        for (column_index, columns) in column_axes.iter().copied().enumerate() {
            if row_index == 0 && column_index == 0 {
                continue;
            }
            let candidate =
                assign_items_to_axes(table, source_items, row_count, column_count, rows, columns);
            let score = raw_cell_match_score(&candidate.0, &table.cells, geometry);
            if score > best_score
                && candidate.1
                && raw_grid_is_fully_corroborated(&candidate.0, &table.cells, geometry)
            {
                best = candidate;
                best_score = score;
                selected_axes = (rows, columns);
            }
        }
    }

    // Heuristic detectors cluster and assign columns from each item's leading
    // edge, while structured capture defaults to centers so ruled-grid and
    // centered values retain their established geometry. A fully corroborated
    // row can prove that a long label crossed an anchor midpoint; resolve only
    // that row and leave unrelated raw-source gaps untouched.
    if best.1 && matches!(selected_axes.1, TableAxis::Anchors(_)) {
        let mut leading = assign_items_to_axes_with_column_anchor(
            table,
            source_items,
            row_count,
            column_count,
            selected_axes.0,
            selected_axes.1,
            ColumnItemAnchor::Leading,
        );
        for row in 0..row_count {
            if row_has_complete_leading_column_association(
                table,
                source_items,
                row,
                column_count,
                selected_axes.0,
                selected_axes.1,
            ) && raw_row_is_fully_corroborated(
                &leading.0[row],
                table.cells.get(row),
                column_count,
                geometry,
            ) && row_only_moves_from_confirmed_blank_origins(
                &best.0[row],
                &leading.0[row],
                table.cells.get(row),
                column_count,
            ) && raw_row_match_score(
                &leading.0[row],
                table.cells.get(row),
                column_count,
                geometry,
            ) > raw_row_match_score(
                &best.0[row],
                table.cells.get(row),
                column_count,
                geometry,
            ) {
                best.0[row] = std::mem::take(&mut leading.0[row]);
            }
        }
    }
    (best.0, best.1)
}

fn assign_items_to_axes(
    table: &Table,
    source_items: &[TextItem],
    row_count: usize,
    column_count: usize,
    rows: TableAxis<'_>,
    columns: TableAxis<'_>,
) -> (Vec<Vec<Vec<TextItem>>>, bool) {
    assign_items_to_axes_with_column_anchor(
        table,
        source_items,
        row_count,
        column_count,
        rows,
        columns,
        ColumnItemAnchor::Center,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColumnItemAnchor {
    Center,
    Leading,
}

fn assign_items_to_axes_with_column_anchor(
    table: &Table,
    source_items: &[TextItem],
    row_count: usize,
    column_count: usize,
    rows: TableAxis<'_>,
    columns: TableAxis<'_>,
    column_anchor: ColumnItemAnchor,
) -> (Vec<Vec<Vec<TextItem>>>, bool) {
    let mut result = vec![vec![Vec::new(); column_count]; row_count];
    let mut reliable = true;
    for &index in &table.item_indices {
        let Some(item) = source_items.get(index) else {
            reliable = false;
            continue;
        };
        let row = axis_index(item.y, rows);
        let column_x = match column_anchor {
            ColumnItemAnchor::Center => item.x + item.width / 2.0,
            ColumnItemAnchor::Leading => item.x,
        };
        let column = axis_index(column_x, columns);
        if let (Some(row), Some(column)) = (row, column) {
            if row < row_count && column < column_count {
                result[row][column].push(item.clone());
            } else {
                reliable = false;
            }
        } else {
            reliable = false;
        }
    }
    (result, reliable)
}

/// An alternative interpretation of ambiguous N-coordinate geometry must
/// reconstruct every established raw slot and leave blank/ragged slots empty.
/// A count alone is unsafe when duplicate values let one shifted row appear to
/// match while a different source line lands in a known blank cell.
fn raw_grid_is_fully_corroborated(
    cell_items: &[Vec<Vec<TextItem>>],
    raw_cells: &[Vec<String>],
    geometry: PageGeometry,
) -> bool {
    cell_items.iter().enumerate().all(|(row, cells)| {
        cells.iter().enumerate().all(|(column, items)| {
            let raw_text = raw_cells
                .get(row)
                .and_then(|raw_row| raw_row.get(column))
                .map(String::as_str)
                .unwrap_or("");
            if raw_text.trim().is_empty() {
                items.is_empty()
            } else {
                lines_matching_raw_cell(&source_lines(items, geometry), raw_text)
            }
        })
    })
}

/// A row-local leading-edge interpretation is safe only when every claimed
/// source item for that row still maps to a known column. Items whose row is
/// unknown make the entire alternative unsafe rather than being guessed into
/// a nearby cell.
fn row_has_complete_leading_column_association(
    table: &Table,
    source_items: &[TextItem],
    row: usize,
    column_count: usize,
    rows: TableAxis<'_>,
    columns: TableAxis<'_>,
) -> bool {
    table.item_indices.iter().all(|&index| {
        let Some(item) = source_items.get(index) else {
            return false;
        };
        match axis_index(item.y, rows) {
            Some(item_row) if item_row == row => {
                axis_index(item.x, columns).is_some_and(|column| column < column_count)
            }
            Some(_) => true,
            None => false,
        }
    })
}

/// A row alternative cannot rely on a raw source string that has no cell slot.
/// Every logical column must be present, matched exactly when nonblank, and
/// empty when the detector established it as blank.
fn raw_row_is_fully_corroborated(
    cell_items: &[Vec<TextItem>],
    raw_cells: Option<&Vec<String>>,
    column_count: usize,
    geometry: PageGeometry,
) -> bool {
    let Some(raw_cells) = raw_cells else {
        return false;
    };
    (0..column_count).all(|column| {
        let Some(raw_text) = raw_cells.get(column) else {
            return false;
        };
        let items = cell_items
            .get(column)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if raw_text.trim().is_empty() {
            items.is_empty()
        } else {
            lines_matching_raw_cell(&source_lines(items, geometry), raw_text)
        }
    })
}

fn raw_row_match_score(
    cell_items: &[Vec<TextItem>],
    raw_cells: Option<&Vec<String>>,
    column_count: usize,
    geometry: PageGeometry,
) -> usize {
    let Some(raw_cells) = raw_cells else {
        return 0;
    };
    (0..column_count)
        .filter(|&column| {
            raw_cells.get(column).is_some_and(|raw_text| {
                !raw_text.trim().is_empty()
                    && cell_items.get(column).is_some_and(|items| {
                        lines_matching_raw_cell(&source_lines(items, geometry), raw_text)
                    })
            })
        })
        .count()
}

/// An overflow reassociation may only move literal source evidence out of a
/// detector-confirmed blank. A nonblank raw slot with existing source items is
/// already an established association, even if another interpretation would
/// score higher, so leave it unchanged.
fn row_only_moves_from_confirmed_blank_origins(
    centered_items: &[Vec<TextItem>],
    leading_items: &[Vec<TextItem>],
    raw_cells: Option<&Vec<String>>,
    column_count: usize,
) -> bool {
    let Some(raw_cells) = raw_cells else {
        return false;
    };
    let mut changed = false;
    for column in 0..column_count {
        let Some(raw_text) = raw_cells.get(column) else {
            return false;
        };
        let centered = centered_items
            .get(column)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let leading = leading_items
            .get(column)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if same_source_items(centered, leading) {
            continue;
        }
        changed = true;
        if raw_text.trim().is_empty() {
            if !leading.is_empty() {
                return false;
            }
        } else if !centered.is_empty() {
            return false;
        }
    }
    changed
}

fn same_source_items(left: &[TextItem], right: &[TextItem]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.text == right.text
                && left.x.to_bits() == right.x.to_bits()
                && left.y.to_bits() == right.y.to_bits()
                && left.width.to_bits() == right.width.to_bits()
                && left.height.to_bits() == right.height.to_bits()
                && left.page == right.page
                && left.mcid == right.mcid
        })
}

fn raw_cell_match_score(
    cell_items: &[Vec<Vec<TextItem>>],
    raw_cells: &[Vec<String>],
    geometry: PageGeometry,
) -> usize {
    let mut score = 0;
    for (row, raw_row) in raw_cells.iter().enumerate() {
        for (column, raw_text) in raw_row.iter().enumerate() {
            if raw_text.trim().is_empty() {
                continue;
            }
            let matches = cell_items
                .get(row)
                .and_then(|cells| cells.get(column))
                .is_some_and(|items| {
                    lines_matching_raw_cell(&source_lines(items, geometry), raw_text)
                });
            if matches {
                score += 1;
            }
        }
    }
    score
}

/// A detector's coordinates are either one position per raw logical slot or
/// one more edge than slots. The raw `cells` matrix remains authoritative for
/// public dimensions; this only interprets known detector geometry for source
/// association. Any other shape is unresolved rather than guessed.
#[derive(Clone, Copy)]
enum TableAxis<'a> {
    Anchors(&'a [f32]),
    Edges(&'a [f32]),
    LeadingEdges {
        positions: &'a [f32],
        trailing_edge: f32,
    },
}

fn table_axis(values: &[f32], logical_count: usize) -> Option<TableAxis<'_>> {
    let strictly_monotonic = values.windows(2).all(|pair| pair[1] > pair[0])
        || values.windows(2).all(|pair| pair[1] < pair[0]);
    if values.len() == logical_count
        && values.iter().all(|value| value.is_finite())
        && strictly_monotonic
    {
        return Some(TableAxis::Anchors(values));
    }
    if values.len() == logical_count.checked_add(1)?
        && values.iter().all(|value| value.is_finite())
        && strictly_monotonic
    {
        return Some(TableAxis::Edges(values));
    }
    None
}

fn table_axis_candidates(
    values: &[f32],
    logical_count: usize,
    supports_leading_edges: bool,
) -> Option<Vec<TableAxis<'_>>> {
    let default_axis = table_axis(values, logical_count)?;
    let mut axes = vec![default_axis];
    if supports_leading_edges && matches!(default_axis, TableAxis::Anchors(_)) {
        if let Some(trailing_edge) = inferred_trailing_edge(values) {
            axes.push(TableAxis::LeadingEdges {
                positions: values,
                trailing_edge,
            });
        }
    }
    Some(axes)
}

fn inferred_trailing_edge(values: &[f32]) -> Option<f32> {
    let last = *values.last()?;
    let previous = *values.get(values.len().checked_sub(2)?)?;
    let trailing_edge = last + (last - previous);
    (trailing_edge.is_finite() && trailing_edge != last).then_some(trailing_edge)
}

fn axis_index(value: f32, axis: TableAxis<'_>) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    match axis {
        TableAxis::Anchors(values) => nearest_index(value, values),
        TableAxis::Edges(values) => values.windows(2).position(|pair| {
            let low = pair[0].min(pair[1]);
            let high = pair[0].max(pair[1]);
            value >= low - 2.0 && value <= high + 2.0
        }),
        TableAxis::LeadingEdges {
            positions,
            trailing_edge,
        } => positions
            .iter()
            .enumerate()
            .find_map(|(index, &leading_edge)| {
                let following_edge = positions.get(index + 1).copied().unwrap_or(trailing_edge);
                let low = leading_edge.min(following_edge);
                let high = leading_edge.max(following_edge);
                (value >= low - 2.0 && value <= high + 2.0).then_some(index)
            }),
    }
}

fn nearest_index(value: f32, values: &[f32]) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    values
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.is_finite())
        .min_by(|(_, left), (_, right)| (value - **left).abs().total_cmp(&(value - **right).abs()))
        .map(|(index, _)| index)
}

fn normalized_spans(spans: &[TableSpan], row_count: usize, column_count: usize) -> Vec<TableSpan> {
    let mut spans = spans
        .iter()
        .copied()
        .filter(|span| span_fits_grid(*span, row_count, column_count))
        .collect::<Vec<_>>();
    spans.sort_by_key(|span| (span.row, span.column));
    spans.dedup_by_key(|span| (span.row, span.column));
    let mut occupied = HashSet::new();
    let mut accepted = Vec::new();
    for span in spans {
        let overlaps = (span.row..span.row + span.row_span).any(|row| {
            (span.column..span.column + span.column_span)
                .any(|column| occupied.contains(&(row, column)))
        });
        if overlaps {
            continue;
        }
        for row in span.row..span.row + span.row_span {
            for column in span.column..span.column + span.column_span {
                occupied.insert((row, column));
            }
        }
        accepted.push(span);
    }
    accepted
}

/// Table spans are detector evidence, not presentation hints. A malformed
/// span must be omitted rather than clipped into a different public span.
fn span_fits_grid(span: TableSpan, row_count: usize, column_count: usize) -> bool {
    if span.row_span == 0 || span.column_span == 0 {
        return false;
    }
    let Some(row_end) = span.row.checked_add(span.row_span) else {
        return false;
    };
    let Some(column_end) = span.column.checked_add(span.column_span) else {
        return false;
    };
    row_end <= row_count && column_end <= column_count
}

fn cell_role(
    items: &[TextItem],
    struct_roles: Option<&HashMap<u32, HashMap<i64, StructRole>>>,
) -> StructuredCellRole {
    let Some(struct_roles) = struct_roles else {
        return StructuredCellRole::Unknown;
    };
    let mut has_body = false;
    for item in items {
        let Some(mcid) = item.mcid else {
            continue;
        };
        match struct_roles
            .get(&item.page)
            .and_then(|roles| roles.get(&mcid))
        {
            Some(StructRole::TH) => return StructuredCellRole::Header,
            Some(StructRole::TD) => has_body = true,
            _ => {}
        }
    }
    if has_body {
        StructuredCellRole::Body
    } else {
        StructuredCellRole::Unknown
    }
}

fn bounds_for_items(items: &[TextItem], geometry: PageGeometry) -> Option<StructuredBounds> {
    union_bounds(
        items
            .iter()
            .filter_map(|item| bounds_for_rect(geometry, item.x, item.y, item.width, item.height)),
    )
}

fn union_bounds(bounds: impl Iterator<Item = StructuredBounds>) -> Option<StructuredBounds> {
    let mut result: Option<(f32, f32, f32, f32)> = None;
    for bounds in bounds {
        let x1 = bounds.x + bounds.width;
        let y1 = bounds.y + bounds.height;
        if !(bounds.x.is_finite() && bounds.y.is_finite() && x1.is_finite() && y1.is_finite()) {
            continue;
        }
        result = Some(match result {
            Some((x0, y0, right, bottom)) => (
                x0.min(bounds.x),
                y0.min(bounds.y),
                right.max(x1),
                bottom.max(y1),
            ),
            None => (bounds.x, bounds.y, x1, y1),
        });
    }
    result.map(|(x, y, right, bottom)| StructuredBounds {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

fn bounds_for_rect(
    geometry: PageGeometry,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) -> Option<StructuredBounds> {
    if !geometry.bounds_available {
        return None;
    }
    if !(x.is_finite() && y.is_finite() && width.is_finite() && height.is_finite()) {
        return None;
    }
    let left = x.min(x + width).max(geometry.x0);
    let right = x.max(x + width).min(geometry.x1);
    let bottom = y.min(y + height).max(geometry.y0);
    let top = y.max(y + height).min(geometry.y1);
    if !(left.is_finite()
        && right.is_finite()
        && bottom.is_finite()
        && top.is_finite()
        && right >= left
        && top >= bottom)
    {
        return None;
    }
    if right == left || top == bottom {
        return None;
    }
    let corners = [
        transform_point(geometry, left, bottom),
        transform_point(geometry, left, top),
        transform_point(geometry, right, bottom),
        transform_point(geometry, right, top),
    ];
    let x0 = corners.iter().map(|point| point.0).reduce(f32::min)?;
    let y0 = corners.iter().map(|point| point.1).reduce(f32::min)?;
    let x1 = corners.iter().map(|point| point.0).reduce(f32::max)?;
    let y1 = corners.iter().map(|point| point.1).reduce(f32::max)?;
    // `/UserUnit` scales the public point size but cancels out when geometry
    // is normalized against that same visible page.
    let width = geometry.upright_source_width();
    let height = geometry.upright_source_height();
    if !(x0.is_finite()
        && y0.is_finite()
        && x1.is_finite()
        && y1.is_finite()
        && width.is_finite()
        && height.is_finite()
        && width > 0.0
        && height > 0.0)
    {
        return None;
    }
    Some(StructuredBounds {
        x: (x0 / width).clamp(0.0, 1.0),
        y: (y0 / height).clamp(0.0, 1.0),
        width: ((x1 - x0) / width).clamp(0.0, 1.0),
        height: ((y1 - y0) / height).clamp(0.0, 1.0),
    })
}

fn transform_point(geometry: PageGeometry, x: f32, y: f32) -> (f32, f32) {
    let x = x - geometry.x0;
    let y = y - geometry.y0;
    match geometry.rotation {
        0 => (x, geometry.source_height() - y),
        90 => (y, x),
        180 => (geometry.source_width() - x, y),
        270 => (geometry.source_height() - y, geometry.source_width() - x),
        _ => unreachable!("page rotation is normalized before use"),
    }
}

fn page_geometry(document: &Document, page_id: ObjectId) -> Result<PageGeometry, PdfError> {
    let visible_box = visible_page_box(document, page_id).ok_or(PdfError::InvalidStructure)?;
    let (x0, y0, x1, y1) = (
        visible_box.x0,
        visible_box.y0,
        visible_box.x1,
        visible_box.y1,
    );
    let (source_width, source_height) = (visible_box.width(), visible_box.height());
    let rotation = inherited_object(document, page_id, b"Rotate")
        .as_ref()
        .and_then(|object| resolved_number(document, object))
        .map(|rotation| rotation.round() as i32)
        .unwrap_or(0)
        .rem_euclid(360);
    let rotation = match rotation {
        0 | 90 | 180 | 270 => rotation as u16,
        _ => return Err(PdfError::InvalidStructure),
    };
    let user_unit = inherited_object(document, page_id, b"UserUnit")
        .as_ref()
        .and_then(|object| resolved_number(document, object))
        .unwrap_or(1.0);
    if !user_unit.is_finite() || user_unit <= 0.0 {
        return Err(PdfError::InvalidStructure);
    }
    let size = if matches!(rotation, 90 | 270) {
        StructuredPageSize {
            width: source_height * user_unit,
            height: source_width * user_unit,
        }
    } else {
        StructuredPageSize {
            width: source_width * user_unit,
            height: source_height * user_unit,
        }
    };
    if !(size.width.is_finite() && size.height.is_finite() && size.width > 0.0 && size.height > 0.0)
    {
        return Err(PdfError::InvalidStructure);
    }
    Ok(PageGeometry {
        x0,
        y0,
        x1,
        y1,
        rotation,
        size,
        bounds_available: true,
    })
}

fn inherited_object(document: &Document, page_id: ObjectId, key: &[u8]) -> Option<Object> {
    let mut current = page_id;
    for _ in 0..32 {
        let dictionary = document.get_dictionary(current).ok()?;
        if let Ok(value) = dictionary.get(key) {
            return Some(value.clone());
        }
        current = match dictionary.get(b"Parent") {
            Ok(Object::Reference(parent)) => *parent,
            _ => return None,
        };
    }
    None
}

fn number(object: &Object) -> Option<f32> {
    match object {
        Object::Integer(value) => Some(*value as f32),
        Object::Real(value) => Some(*value),
        _ => None,
    }
}

fn resolved_number(document: &Document, object: &Object) -> Option<f32> {
    match object {
        Object::Reference(reference) => document.get_object(*reference).ok().and_then(number),
        object => number(object),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ItemType;

    fn geometry(rotation: u16) -> PageGeometry {
        PageGeometry {
            x0: 10.0,
            y0: 20.0,
            x1: 110.0,
            y1: 220.0,
            rotation,
            size: if matches!(rotation, 90 | 270) {
                StructuredPageSize {
                    width: 200.0,
                    height: 100.0,
                }
            } else {
                StructuredPageSize {
                    width: 100.0,
                    height: 200.0,
                }
            },
            bounds_available: true,
        }
    }

    #[test]
    fn normalizes_an_offset_crop_box() {
        assert_eq!(
            bounds_for_rect(geometry(0), 20.0, 40.0, 20.0, 40.0),
            Some(StructuredBounds {
                x: 0.1,
                y: 0.7,
                width: 0.2,
                height: 0.2,
            })
        );
    }

    #[test]
    fn rotates_crop_relative_bounds_to_an_upright_top_left_space() {
        assert_eq!(
            bounds_for_rect(geometry(90), 10.0, 20.0, 20.0, 40.0),
            Some(StructuredBounds {
                x: 0.0,
                y: 0.0,
                width: 0.2,
                height: 0.2,
            })
        );
    }

    #[test]
    fn resolves_an_inherited_offset_crop_box_and_rotation() {
        use lopdf::{dictionary, Document, Object};

        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let page_id = document.new_object_id();
        document.objects.insert(
            pages_id,
            dictionary! {
                "Type" => "Pages",
                "CropBox" => vec![
                    Object::Integer(20),
                    Object::Integer(30),
                    Object::Integer(180),
                    Object::Integer(270),
                ],
                "Rotate" => 90,
                "UserUnit" => 2,
            }
            .into(),
        );
        document.objects.insert(
            page_id,
            dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
            }
            .into(),
        );

        let geometry = page_geometry(&document, page_id).expect("inherited page geometry");
        assert_eq!(geometry.size.width, 480.0);
        assert_eq!(geometry.size.height, 320.0);
        assert_eq!(
            bounds_for_rect(geometry, 20.0, 30.0, 16.0, 24.0),
            Some(StructuredBounds {
                x: 0.0,
                y: 0.0,
                width: 0.1,
                height: 0.1,
            })
        );
    }

    #[test]
    fn clips_geometry_without_inventing_off_page_bounds() {
        assert_eq!(bounds_for_rect(geometry(0), -100.0, -100.0, 5.0, 5.0), None);
        let clipped = bounds_for_rect(geometry(0), 0.0, 0.0, 30.0, 40.0).unwrap();
        assert_eq!(clipped.x, 0.0);
        assert_eq!(clipped.y, 0.9);
        assert_eq!(clipped.width, 0.2);
        assert_eq!(clipped.height, 0.1);
    }

    #[test]
    fn omits_bounds_when_content_rotation_has_no_proven_page_transform() {
        let mut page_geometry = geometry(0);
        page_geometry.disable_bounds();
        assert_eq!(bounds_for_rect(page_geometry, 20.0, 40.0, 20.0, 40.0), None);
    }

    fn item(text: &str, x: f32, y: f32) -> TextItem {
        TextItem {
            text: text.to_owned(),
            x,
            y,
            width: 40.0,
            height: 10.0,
            rotation: 0.0,
            advance_known: true,
            font: "F1".to_owned(),
            font_tag: String::new(),
            legacy_symbol_rewrite: false,
            font_size: 10.0,
            page: 1,
            is_bold: false,
            is_italic: false,
            font_weight: None,
            bold_source: None,
            fixed_pitch: None,
            fill_color: None,
            stroke_color: None,
            render_mode: None,
            is_underline: false,
            is_strikeout: false,
            item_type: ItemType::Text,
            mcid: None,
            baseline_shift: 0.0,
        }
    }

    #[test]
    fn captures_raw_table_rows_before_markdown_cleanup() {
        let items = vec![
            item("Date", 50.0, 500.0),
            item("Description", 150.0, 500.0),
            item("03 Jan", 50.0, 480.0),
            item("DUPLICATE", 150.0, 480.0),
            item("04 Jan", 50.0, 460.0),
            item("DUPLICATE", 150.0, 460.0),
            item("05 Jan", 50.0, 420.0),
            item("WRAPPED", 150.0, 420.0),
            item("CONTINUED", 150.0, 400.0),
        ];
        let table = Table::new(
            vec![50.0, 150.0],
            vec![500.0, 480.0, 460.0, 440.0, 420.0, 400.0],
            vec![
                vec!["Date".into(), "Description".into()],
                vec!["03 Jan".into(), "DUPLICATE".into()],
                vec!["04 Jan".into(), "DUPLICATE".into()],
                vec!["".into(), "".into()],
                vec!["05 Jan".into(), "WRAPPED".into()],
                vec!["".into(), "CONTINUED".into()],
            ],
            (0..items.len()).collect(),
        );

        let StructuredBlockContent::Table {
            row_count,
            column_count,
            cells,
            ..
        } = table_block(&table, &items, geometry(0), None).content
        else {
            panic!("expected a table block");
        };

        assert_eq!((row_count, column_count), (6, 2));
        assert_eq!(cells.len(), 12, "every unspanned raw grid slot remains");
        assert_eq!(
            cells
                .iter()
                .filter(|cell| cell.lines.iter().any(|line| line.text == "DUPLICATE"))
                .count(),
            2,
            "identical source rows must remain distinct"
        );
        assert!(cells.iter().any(|cell| {
            cell.row == 3
                && cell.column == 0
                && cell.state == StructuredCellState::Empty
                && cell.lines.is_empty()
        }));
        assert!(cells.iter().any(|cell| {
            cell.row == 5
                && cell.column == 1
                && cell
                    .lines
                    .iter()
                    .map(|line| line.text.as_str())
                    .collect::<Vec<_>>()
                    == ["CONTINUED"]
        }));

        let markdown = crate::tables::table_to_markdown(&table);
        assert!(markdown.contains("WRAPPED CONTINUED"));
        assert!(
            !markdown.contains("| |\n"),
            "Markdown cleanup removes the empty presentation row; structured output does not"
        );
    }

    #[test]
    fn raw_cell_line_matching_keeps_original_line_boundaries() {
        let lines = vec![
            StructuredTextLine {
                text: "First  line".into(),
                bounds: None,
            },
            StructuredTextLine {
                text: "Second line".into(),
                bounds: None,
            },
        ];
        assert!(lines_matching_raw_cell(&[], ""));
        assert!(!lines_matching_raw_cell(&lines, ""));
        assert!(lines_matching_raw_cell(&lines, "First\nline\tSecond line"));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "First  line");
        assert_eq!(lines[1].text, "Second line");
    }

    #[test]
    fn unresolved_raw_grid_text_keeps_positioned_lines_separate() {
        let items = vec![item("Observed source", 50.0, 100.0)];
        let table = Table::new(
            vec![50.0],
            vec![100.0],
            vec![vec!["Detector-normalized text".into()]],
            vec![0],
        );
        let StructuredBlockContent::Table { cells, .. } =
            table_block(&table, &items, geometry(0), None).content
        else {
            panic!("expected a table block");
        };

        let cell = &cells[0];
        assert_eq!(cell.state, StructuredCellState::Unresolved);
        assert_eq!(cell.lines.len(), 1);
        assert_eq!(cell.lines[0].text, "Observed source");
        assert_eq!(
            cell.detected_text.as_deref(),
            Some("Detector-normalized text")
        );
        assert!(cell.bounds.is_some());
    }

    #[test]
    fn vector_style_edge_geometry_does_not_expand_a_two_by_two_raw_grid() {
        // Some vector strategies retain N + 1 grid edges. The raw `cells`
        // matrix remains the logical grid, so it must not gain a phantom row
        // or column from those edges.
        let table = Table::new(
            vec![0.0, 100.0, 200.0],
            vec![200.0, 100.0, 0.0],
            vec![
                vec!["A1".into(), "B1".into()],
                vec!["A2".into(), "B2".into()],
            ],
            Vec::new(),
        );
        let StructuredBlockContent::Table {
            row_count,
            column_count,
            cells,
            ..
        } = table_block(&table, &[], geometry(0), None).content
        else {
            panic!("expected a table block");
        };
        assert_eq!((row_count, column_count), (2, 2));
        assert_eq!(cells.len(), 4);
        assert!(cells
            .iter()
            .all(|cell| cell.state == StructuredCellState::Unresolved));
    }

    #[test]
    fn ruled_grid_edges_associate_source_items_without_expanding_the_raw_grid() {
        let items = vec![
            item("A1", 20.0, 500.0),
            item("B1", 120.0, 500.0),
            item("A2", 20.0, 480.0),
            item("B2", 120.0, 480.0),
        ];
        // Line-based ruled-grid detection retains N + 1 X edges, while the
        // raw matrix has N columns. It may retain row anchors instead of the
        // final bottom edge, so rows remain exactly N here.
        let table = Table::new(
            vec![0.0, 100.0, 200.0],
            vec![500.0, 480.0],
            vec![
                vec!["A1".into(), "B1".into()],
                vec!["A2".into(), "B2".into()],
            ],
            (0..items.len()).collect(),
        );
        let StructuredBlockContent::Table {
            row_count,
            column_count,
            cells,
            ..
        } = table_block(&table, &items, test_page_geometry(), None).content
        else {
            panic!("expected a table block");
        };

        assert_eq!((row_count, column_count), (2, 2));
        assert_eq!(cells.len(), 4);
        assert!(cells.iter().all(|cell| {
            cell.state == StructuredCellState::Extracted
                && cell.lines.len() == 1
                && cell.bounds.is_some()
        }));
    }

    #[test]
    fn ambiguous_geometry_never_confirms_blank_raw_slots() {
        let table = Table::new(
            // Four positions cannot describe a two-column anchor or edge
            // axis, so association is deliberately disabled.
            vec![0.0, 50.0, 100.0, 150.0],
            vec![500.0, 480.0],
            vec![vec!["Known raw text".into(), String::new()]],
            Vec::new(),
        );
        let StructuredBlockContent::Table { cells, .. } =
            table_block(&table, &[], test_page_geometry(), None).content
        else {
            panic!("expected a table block");
        };

        let blank_slot = cells
            .iter()
            .find(|cell| cell.row == 0 && cell.column == 1)
            .expect("raw blank slot");
        assert_eq!(blank_slot.state, StructuredCellState::Unresolved);
    }

    #[test]
    fn non_monotonic_anchor_geometry_disables_table_item_association() {
        let items = vec![
            item("A", 20.0, 500.0),
            item("B", 120.0, 500.0),
            item("C", 220.0, 500.0),
        ];
        let table = Table::new(
            // These are logical anchors, not edges. They are finite but do
            // not describe a direction, so nearest-index assignment would be
            // ambiguous evidence rather than a source location.
            vec![20.0, 220.0, 120.0],
            vec![500.0],
            vec![vec!["A".into(), "B".into(), "C".into()]],
            (0..items.len()).collect(),
        );
        let StructuredBlockContent::Table { cells, .. } =
            table_block(&table, &items, geometry(0), None).content
        else {
            panic!("expected a table block");
        };

        assert!(cells.iter().all(|cell| {
            cell.state == StructuredCellState::Unresolved && cell.lines.is_empty()
        }));
        assert!(table_axis(&[20.0, 20.0], 2).is_none());
        assert!(table_axis(&[220.0, 120.0, 20.0], 3).is_some());
    }

    #[test]
    fn ragged_raw_rows_never_turn_missing_slots_into_confirmed_blanks() {
        let items = vec![
            item("A", 50.0, 500.0),
            item("B", 150.0, 500.0),
            item("C", 50.0, 480.0),
        ];
        let table = Table::new(
            vec![50.0, 150.0],
            vec![500.0, 480.0],
            vec![vec!["A".into(), "B".into()], vec!["C".into()]],
            (0..items.len()).collect(),
        );
        let StructuredBlockContent::Table { cells, .. } =
            table_block(&table, &items, geometry(0), None).content
        else {
            panic!("expected a table block");
        };

        let missing_slot = cells
            .iter()
            .find(|cell| cell.row == 1 && cell.column == 1)
            .expect("public grid includes the ragged row's missing slot");
        assert_eq!(missing_slot.state, StructuredCellState::Unresolved);
        assert!(missing_slot.lines.is_empty());
    }

    #[test]
    fn emits_only_the_anchor_for_detector_supported_row_spans() {
        let items = vec![
            item("Merged", 50.0, 500.0),
            item("Right top", 150.0, 500.0),
            item("Right bottom", 150.0, 480.0),
        ];
        let table = Table::new(
            vec![50.0, 150.0],
            vec![500.0, 480.0],
            vec![
                vec!["Merged".into(), "Right top".into()],
                vec!["".into(), "Right bottom".into()],
            ],
            (0..items.len()).collect(),
        )
        .with_spans(vec![TableSpan {
            row: 0,
            column: 0,
            row_span: 2,
            column_span: 1,
        }]);

        let StructuredBlockContent::Table { cells, .. } =
            table_block(&table, &items, geometry(0), None).content
        else {
            panic!("expected a table block");
        };

        let anchor = cells
            .iter()
            .find(|cell| cell.row == 0 && cell.column == 0)
            .expect("span anchor");
        assert_eq!((anchor.row_span, anchor.column_span), (2, 1));
        assert!(
            !cells.iter().any(|cell| cell.row == 1 && cell.column == 0),
            "covered slots are represented by the anchor span, never invented blanks"
        );
        assert_eq!(cells.len(), 3);
    }

    #[test]
    fn span_evidence_is_preserved_or_skipped_without_rewriting_it() {
        let valid = TableSpan {
            row: 0,
            column: 0,
            row_span: 2,
            column_span: 1,
        };
        let valid_horizontal = TableSpan {
            row: 0,
            column: 1,
            row_span: 1,
            column_span: 2,
        };
        let normalized = normalized_spans(
            &[
                valid,
                valid_horizontal,
                // This conflicts with `valid`, so it cannot describe a
                // second non-overlapping public anchor.
                TableSpan {
                    row: 1,
                    column: 0,
                    row_span: 1,
                    column_span: 2,
                },
                // These malformed values used to be clamped into smaller
                // public spans. They must now disappear entirely.
                TableSpan {
                    row: 1,
                    column: 1,
                    row_span: 2,
                    column_span: 1,
                },
                TableSpan {
                    row: 0,
                    column: 2,
                    row_span: 1,
                    column_span: 0,
                },
                TableSpan {
                    row: usize::MAX,
                    column: 0,
                    row_span: 1,
                    column_span: 1,
                },
            ],
            2,
            3,
        );

        assert_eq!(normalized, vec![valid, valid_horizontal]);
    }

    #[test]
    fn invalid_span_evidence_keeps_raw_grid_slots_explicit() {
        let table = Table::new(
            vec![50.0, 150.0],
            vec![500.0, 480.0],
            vec![vec!["A".into(), "B".into()], vec!["C".into(), "D".into()]],
            Vec::new(),
        )
        .with_spans(vec![TableSpan {
            row: 1,
            column: 0,
            row_span: 2,
            column_span: 1,
        }]);

        let StructuredBlockContent::Table { cells, .. } =
            table_block(&table, &[], geometry(0), None).content
        else {
            panic!("expected a table block");
        };

        assert_eq!(cells.len(), 4);
        assert!(cells
            .iter()
            .all(|cell| (cell.row_span, cell.column_span) == (1, 1)));
    }

    #[test]
    fn document_page_coverage_keeps_selected_source_indexes_without_text() {
        let mut geometries = BTreeMap::new();
        geometries.insert(1, geometry(0));
        geometries.insert(3, geometry(0));
        let mut statuses = HashMap::new();
        statuses.insert(3, StructuredPageStatus::NeedsOcr);

        let document = document_from_blocks(&geometries, BTreeMap::new(), &statuses);
        assert_eq!(document.pages.len(), 2);
        assert_eq!(document.pages[0].page_index, 0);
        assert_eq!(document.pages[0].status, StructuredPageStatus::Extracted);
        assert!(document.pages[0].blocks.is_empty());
        assert_eq!(document.pages[1].page_index, 2);
        assert_eq!(document.pages[1].status, StructuredPageStatus::NeedsOcr);
        assert!(document.pages[1].blocks.is_empty());
    }
}

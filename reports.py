import base64
import html
import textwrap
from datetime import datetime
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class ReportRenderer:
    def __init__(self, display_table, display_frame, model_label_for_run, preprocessing_details, recommendation_report_frame):
        self.display_table = display_table
        self.display_frame = display_frame
        self.model_label_for_run = model_label_for_run
        self.preprocessing_details = preprocessing_details
        self.recommendation_report_frame = recommendation_report_frame

    def csv_report_table(self, csv_data, css_class="report-table"):
        if not csv_data:
            return ""
        try:
            return self.display_table(pd.read_csv(StringIO(csv_data)), index=False, border=0, classes=css_class)
        except Exception:
            return f"<pre>{html.escape(csv_data)}</pre>"

    def metrics_report_table(self, metrics):
        if not metrics:
            return ""
        rows = [{"Metric": item.get("label", ""), "Value": item.get("value", "")} for item in metrics]
        return self.display_table(pd.DataFrame(rows), index=False, border=0, classes="report-table")

    def report_model_labels(self, tab_name, models):
        tab = {"form_name": tab_name}
        return ", ".join(self.model_label_for_run(tab, model) for model in models)

    def report_metadata_rows(self, tab_name, snapshot, dataset):
        data = dataset.get("data") if dataset else None
        history_entry = snapshot.get("history_entry") or {}
        detail_model = snapshot.get("selected_model") or snapshot.get("selected_detail_model") or "-"
        cv_folds = snapshot.get("selected_cv_folds")
        preprocessing = self.preprocessing_details(snapshot)
        return [
            {"Field": "Generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Field": "Run time", "Value": history_entry.get("timestamp", "-")},
            {"Field": "Report type", "Value": "Pro classification" if tab_name == "pro_classification" else "Pro regression"},
            {"Field": "Dataset", "Value": dataset.get("filename", "-") if dataset else "-"},
            {"Field": "Rows", "Value": len(data) if data is not None else "-"},
            {"Field": "Columns", "Value": len(data.columns) if data is not None else "-"},
            {"Field": "Target", "Value": snapshot.get("selected_target") or "-"},
            {"Field": "Predictors", "Value": ", ".join(snapshot.get("selected_predictors") or []) or "-"},
            {"Field": "Models compared", "Value": self.report_model_labels(tab_name, snapshot.get("selected_models") or []) or "-"},
            {"Field": "Detail model", "Value": self.model_label_for_run({"form_name": tab_name}, detail_model)},
            {"Field": "Test split", "Value": f"{float(snapshot.get('selected_test_size') or 0):.0%}"},
            {"Field": "Cross-validation", "Value": f"{cv_folds} folds" if cv_folds else "Off"},
            {"Field": "Missing values", "Value": preprocessing.get("Missing values", "-")},
            {"Field": "Categorical encoding", "Value": preprocessing.get("Categorical encoding", "-")},
            {"Field": "Feature scaling", "Value": preprocessing.get("Feature scaling", "-")},
            {"Field": "Split seed", "Value": snapshot.get("selected_split_seed", "-")},
            {"Field": "Outlier handling", "Value": preprocessing.get("Outlier handling", "-")},
            {"Field": "Probability calibration", "Value": {"off": "Off", "sigmoid": "Sigmoid", "isotonic": "Isotonic"}.get(snapshot.get("selected_calibration", "off"), "Off") if tab_name == "pro_classification" else "-"},
            {"Field": "Hyperparameter tuning", "Value": {"off": "Off", "grid": "Grid search", "random": "Random search"}.get(snapshot.get("selected_tuning_mode", "off"), "Off")},
            {"Field": "Random search iterations", "Value": snapshot.get("selected_tuning_iterations", 10)},
            {"Field": "Decision threshold", "Value": f"{float(snapshot.get('selected_threshold') or 0.5):.3f}" if tab_name == "pro_classification" else "-"},
        ]

    def report_metadata_table(self, tab_name, snapshot, dataset):
        rows = self.report_metadata_rows(tab_name, snapshot, dataset)
        return self.display_table(pd.DataFrame(rows), index=False, border=0, classes="report-table")

    def report_image(self, title, image_data, alt_text):
        if not image_data:
            return ""
        return (
            f"<section><h2>{html.escape(title)}</h2>"
            f"<img class=\"report-plot\" src=\"data:image/png;base64,{html.escape(image_data)}\" "
            f"alt=\"{html.escape(alt_text)}\"></section>"
        )

    def pro_report_html(self, tab_name, snapshot, dataset):
        output = snapshot.get("output") or {}
        artifacts = snapshot.get("download_artifacts") or {}
        recommendation = snapshot.get("recommendation")
        report_title = "Pro Classification Report" if tab_name == "pro_classification" else "Pro Regression Report"
        selected_title = output.get("title") or "Selected model results"
        selected_description = output.get("description") or ""
        sections = [
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
            f"<title>{html.escape(report_title)}</title>",
            """
            <style>
              body { color: #1f2933; font-family: Arial, sans-serif; line-height: 1.45; margin: 32px; }
              h1 { font-size: 28px; margin-bottom: 6px; }
              h2 { border-bottom: 1px solid #d8dee8; font-size: 20px; margin-top: 30px; padding-bottom: 6px; }
              h3 { font-size: 16px; margin-top: 22px; }
              .muted { color: #64748b; }
              .report-table { border-collapse: collapse; margin: 10px 0 18px; width: 100%; }
              .report-table th, .report-table td { border: 1px solid #d8dee8; padding: 7px 9px; text-align: left; vertical-align: top; }
              .report-table th { background: #f5f7fa; }
              .report-plot { border: 1px solid #d8dee8; max-width: 100%; }
              pre { background: #f5f7fa; border: 1px solid #d8dee8; overflow: auto; padding: 12px; }
            </style></head><body>
            """,
            f"<h1>{html.escape(report_title)}</h1>",
            f"<p class=\"muted\">Single-run export for the selected Pro model comparison.</p>",
            "<section><h2>Dataset metadata</h2>",
            self.report_metadata_table(tab_name, snapshot, dataset),
            "</section>",
        ]

        comparison_table = self.csv_report_table(artifacts.get("model_comparison"))
        if comparison_table:
            sections.extend(["<section><h2>Model comparison</h2>", comparison_table, "</section>"])

        if recommendation:
            sections.extend(["<section><h2>Explain this run</h2>", self.recommendation_html(recommendation), "</section>"])

        sections.extend([
            "<section><h2>Selected detail model</h2>",
            f"<h3>{html.escape(selected_title)}</h3>",
            f"<p>{html.escape(selected_description)}</p>",
        ])
        metric_table = self.metrics_report_table(output.get("metrics") or [])
        if metric_table:
            sections.extend(["<h3>Metrics</h3>", metric_table])
        cv_summary_table = self.csv_report_table(artifacts.get("cv_summary"))
        cv_diagnostics_table = self.csv_report_table(artifacts.get("cv_diagnostics"))
        if cv_summary_table or cv_diagnostics_table or output.get("cv_plot"):
            sections.extend(["<h3>Cross-validation diagnostics</h3>"])
            if output.get("cv_plot"):
                sections.append(self.report_image("Cross-validation fold scores", output.get("cv_plot"), "Cross-validation fold score plot"))
            if cv_summary_table:
                sections.extend(["<h4>Summary</h4>", cv_summary_table])
            if cv_diagnostics_table:
                sections.extend(["<h4>Fold scores</h4>", cv_diagnostics_table])
        for heading, key in [
            ("Coefficients", "coefficients_html"),
            ("Variable importance", "importances_html"),
            ("Model details", "details_html"),
            ("Confusion matrix", "confusion_html"),
        ]:
            if output.get(key):
                sections.extend([f"<h3>{heading}</h3>", output[key]])
        sections.append("</section>")

        if tab_name == "pro_classification":
            sections.extend(
                [
                    self.report_image("ROC curve", output.get("roc_plot"), "ROC curve"),
                    self.report_image("Precision-recall curve", output.get("pr_plot"), "Precision-recall curve"),
                    self.report_image("Classification tree", output.get("tree_plot"), "Classification tree"),
                ]
            )
            selected_threshold_table = self.csv_report_table(artifacts.get("selected_threshold_metrics"))
            if selected_threshold_table:
                sections.extend(["<section><h2>Selected threshold metrics</h2>", selected_threshold_table, "</section>"])
            threshold_table = self.csv_report_table(artifacts.get("threshold_analysis"))
            if threshold_table:
                sections.extend(["<section><h2>Threshold analysis</h2>", threshold_table, "</section>"])
        else:
            sections.extend(
                [
                    self.report_image("Predicted vs actual", output.get("predicted_actual_plot"), "Predicted vs actual plot"),
                    self.report_image("Residuals vs fitted", output.get("residuals_fitted_plot"), "Residuals vs fitted plot"),
                    self.report_image("Residual distribution", output.get("residual_distribution_plot"), "Residual distribution plot"),
                ]
            )
            residual_table = self.csv_report_table(artifacts.get("residual_diagnostics"))
            if residual_table:
                sections.extend(["<section><h2>Residual diagnostics</h2>", residual_table, "</section>"])

        sections.append("</body></html>")
        return "".join(sections)

    def recommendation_html(self, recommendation):
        frame = self.recommendation_report_frame(recommendation)
        if frame.empty:
            return ""
        return self.display_table(frame, index=False, border=0, classes="report-table")

    def csv_report_frame(self, csv_data):
        if not csv_data:
            return None
        try:
            return pd.read_csv(StringIO(csv_data))
        except Exception:
            return pd.DataFrame({"Value": [csv_data]})

    def output_metrics_frame(self, output):
        metrics = output.get("metrics") or []
        if not metrics:
            return None
        return pd.DataFrame([{"Metric": item.get("label", ""), "Value": item.get("value", "")} for item in metrics])

    def html_table_frame(self, html_table):
        if not html_table:
            return None
        try:
            tables = pd.read_html(StringIO(html_table))
        except (ImportError, ValueError):
            return None
        return tables[0] if tables else None

    def wrap_pdf_value(self, value, width=28):
        text = "" if pd.isna(value) else str(value)
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text

    def pdf_column_wrap_width(self, column_count):
        if column_count <= 2:
            return 44
        if column_count <= 4:
            return 28
        return 20

    def pdf_row_line_count(self, row):
        return max((str(value).count("\n") + 1 for value in row), default=1)

    def pdf_row_height_units(self, line_count):
        return max(1.0, line_count * 1.35 + 0.35)

    def pdf_table_chunks(self, frame, max_line_units):
        chunk_rows = []
        chunk_units = 0
        for _, row in frame.iterrows():
            values = list(row)
            row_units = self.pdf_row_height_units(self.pdf_row_line_count(values))
            if chunk_rows and chunk_units + row_units > max_line_units:
                yield pd.DataFrame(chunk_rows, columns=frame.columns)
                chunk_rows = []
                chunk_units = 0
            chunk_rows.append(values)
            chunk_units += row_units
        if chunk_rows:
            yield pd.DataFrame(chunk_rows, columns=frame.columns)

    def pdf_column_groups(self, frame, max_columns=6):
        columns = list(frame.columns)
        if len(columns) <= max_columns:
            return [columns]

        pinned = [column for column in ["Model", "Status"] if column in columns]
        remaining = [column for column in columns if column not in pinned]
        group_size = max(1, max_columns - len(pinned))
        groups = []
        for start in range(0, len(remaining), group_size):
            groups.append(pinned + remaining[start : start + group_size])
        return groups

    def add_pdf_text_page(self, pdf, title, lines):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.04, 0.94, title, fontsize=20, weight="bold", va="top")
        y = 0.84
        for line in lines:
            wrapped = textwrap.wrap(str(line), width=105) or [""]
            for part in wrapped:
                ax.text(0.05, y, part, fontsize=10.5, va="top")
                y -= 0.04
                if y < 0.08:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis("off")
                    y = 0.94
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_pdf_table_pages(self, pdf, title, frame, rows_per_page=24):
        if frame is None or frame.empty:
            return

        frame = self.display_frame(frame).copy().fillna("")
        column_groups = self.pdf_column_groups(frame)
        prepared_pages = []
        for group_index, columns in enumerate(column_groups, start=1):
            group_frame = frame[columns].copy()
            wrap_width = self.pdf_column_wrap_width(len(group_frame.columns))
            wrapped = group_frame.copy()
            for column in wrapped.columns:
                wrapped[column] = wrapped[column].map(lambda value: self.wrap_pdf_value(value, width=wrap_width))

            chunks = list(self.pdf_table_chunks(wrapped, max_line_units=rows_per_page))
            rendered_rows = 0
            for chunk in chunks:
                prepared_pages.append(
                    self.prepare_pdf_table_block(
                        chunk.copy(),
                        len(group_frame),
                        rendered_rows,
                        len(chunks),
                        group_index,
                        len(column_groups),
                    )
                )
                rendered_rows += len(chunk)

        if len(column_groups) > 1 and self.pdf_blocks_fit_on_single_page(prepared_pages):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.97, title, fontsize=16, weight="bold", va="top")
            current_top = 0.89
            for block in prepared_pages:
                block_title = block["suffix"].strip()
                if block_title:
                    ax.text(0.02, current_top, block_title, fontsize=9.5, weight="bold", color="#475569", va="top")
                    current_top -= 0.032
                self.draw_pdf_table_block(ax, block, current_top)
                current_top -= block["table_height"] + 0.055
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return

        for block in prepared_pages:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.97, f"{title}{block['suffix']}", fontsize=16, weight="bold", va="top")
            self.draw_pdf_table_block(ax, block, 0.89)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def prepare_pdf_table_block(self, chunk, total_rows, rendered_rows, chunk_count, group_index, group_count):
        chunk = chunk.copy()
        row_line_counts = [self.pdf_row_line_count(row) for row in chunk.to_numpy()]
        suffix_parts = []
        if chunk_count > 1:
            suffix_parts.append(f"{rendered_rows + 1}-{rendered_rows + len(chunk)} of {total_rows}")
        if group_count > 1:
            suffix_parts.append(f"columns {group_index} of {group_count}")
        header_units = 1.25
        row_units = [self.pdf_row_height_units(count) for count in row_line_counts]
        total_units = header_units + sum(row_units)
        return {
            "chunk": chunk,
            "suffix": f" ({'; '.join(suffix_parts)})" if suffix_parts else "",
            "header_units": header_units,
            "row_units": row_units,
            "total_units": total_units,
            "table_height": min(0.84, max(0.18, total_units * 0.038)),
        }

    def pdf_blocks_fit_on_single_page(self, blocks):
        if not blocks:
            return False
        needed_height = sum(block["table_height"] for block in blocks)
        needed_height += 0.055 * max(0, len(blocks) - 1)
        needed_height += 0.032 * sum(1 for block in blocks if block["suffix"])
        return needed_height <= 0.82

    def draw_pdf_table_block(self, ax, block, top):
        chunk = block["chunk"]
        table_height = block["table_height"]
        table_bottom = top - table_height
        table = ax.table(
            cellText=chunk.values,
            colLabels=list(chunk.columns),
            cellLoc="left",
            loc="center",
            bbox=[0.02, table_bottom, 0.96, table_height],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.0 if len(chunk.columns) > 6 else 8.2)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(weight="bold")
                cell.set_height(table_height * block["header_units"] / block["total_units"])
            else:
                cell.set_height(table_height * block["row_units"][row - 1] / block["total_units"])
            cell.set_edgecolor("#d8dee8")

    def add_pdf_image_page(self, pdf, title, image_data):
        if not image_data:
            return
        try:
            image = plt.imread(BytesIO(base64.b64decode(image_data)), format="png")
        except Exception:
            return

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.02, 0.97, title, fontsize=16, weight="bold", va="top")
        ax.imshow(image, extent=[0.08, 0.92, 0.08, 0.88], aspect="auto")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def pro_report_pdf_bytes(self, tab_name, snapshot, dataset):
        output = snapshot.get("output") or {}
        artifacts = snapshot.get("download_artifacts") or {}
        recommendation = snapshot.get("recommendation")
        report_title = "Pro Classification Report" if tab_name == "pro_classification" else "Pro Regression Report"
        buffer = BytesIO()

        with PdfPages(buffer) as pdf:
            history_entry = snapshot.get("history_entry") or {}
            self.add_pdf_text_page(
                pdf,
                report_title,
                [
                    "Single-run export for the selected Pro model comparison.",
                    f"Dataset: {(dataset or {}).get('filename', '-')}",
                    f"Target: {snapshot.get('selected_target') or '-'}",
                    f"Detail model: {history_entry.get('detail_model', '-')}",
                    f"Summary: {history_entry.get('summary', '-')}",
                ],
            )
            self.add_pdf_table_pages(pdf, "Dataset Metadata", pd.DataFrame(self.report_metadata_rows(tab_name, snapshot, dataset)), rows_per_page=22)
            self.add_pdf_table_pages(pdf, "Model Comparison", self.csv_report_frame(artifacts.get("model_comparison")), rows_per_page=18)
            self.add_pdf_table_pages(pdf, "Explain This Run", self.recommendation_report_frame(recommendation), rows_per_page=20)
            self.add_pdf_table_pages(pdf, "Selected Detail Metrics", self.output_metrics_frame(output), rows_per_page=24)
            self.add_pdf_image_page(pdf, "Cross-Validation Fold Scores", output.get("cv_plot"))
            self.add_pdf_table_pages(pdf, "Cross-Validation Summary", self.csv_report_frame(artifacts.get("cv_summary")), rows_per_page=24)
            self.add_pdf_table_pages(pdf, "Cross-Validation Fold Scores", self.csv_report_frame(artifacts.get("cv_diagnostics")), rows_per_page=24)

            details_frame_pdf = self.csv_report_frame(artifacts.get("details"))
            if details_frame_pdf is None:
                details_frame_pdf = self.html_table_frame(output.get("details_html"))
            self.add_pdf_table_pages(pdf, "Model Details", details_frame_pdf, rows_per_page=24)
            self.add_pdf_table_pages(pdf, "Coefficients", self.csv_report_frame(artifacts.get("coefficients")), rows_per_page=24)
            self.add_pdf_table_pages(pdf, "Variable Importance", self.csv_report_frame(artifacts.get("variable_importance")), rows_per_page=24)
            self.add_pdf_table_pages(pdf, "Confusion Matrix", self.csv_report_frame(artifacts.get("confusion_matrix")), rows_per_page=24)

            if tab_name == "pro_classification":
                self.add_pdf_image_page(pdf, "ROC Curve", output.get("roc_plot"))
                self.add_pdf_image_page(pdf, "Precision-Recall Curve", output.get("pr_plot"))
                self.add_pdf_image_page(pdf, "Classification Tree", output.get("tree_plot"))
                self.add_pdf_table_pages(pdf, "Selected Threshold Metrics", self.csv_report_frame(artifacts.get("selected_threshold_metrics")), rows_per_page=24)
                self.add_pdf_table_pages(pdf, "Threshold Analysis", self.csv_report_frame(artifacts.get("threshold_analysis")), rows_per_page=24)
            else:
                self.add_pdf_image_page(pdf, "Predicted vs Actual", output.get("predicted_actual_plot"))
                self.add_pdf_image_page(pdf, "Residuals vs Fitted", output.get("residuals_fitted_plot"))
                self.add_pdf_image_page(pdf, "Residual Distribution", output.get("residual_distribution_plot"))
                self.add_pdf_table_pages(pdf, "Residual Diagnostics", self.csv_report_frame(artifacts.get("residual_diagnostics")), rows_per_page=28)

        buffer.seek(0)
        return buffer.getvalue()

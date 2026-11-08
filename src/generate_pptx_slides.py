"""
Generate professional PowerPoint presentation slides for academic project defense / viva.
Builds 10 high-impact slides with modern Navy/Blue academic palette,
clean typography, KPI highlights, comparison tables, and embedded EDA/ML charts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pptx
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from config import REPORTS_DIR, VISUALIZATIONS_DIR

# Academic Color Palette
NAVY_PRIMARY = RGBColor(31, 78, 121)     # #1F4E79
BLUE_ACCENT = RGBColor(37, 99, 235)      # #2563EB
TEXT_DARK = RGBColor(15, 23, 42)         # #0F172A
TEXT_MUTED = RGBColor(100, 116, 139)     # #64748B
BG_LIGHT = RGBColor(248, 250, 252)       # #F8FAFC
CARD_BG = RGBColor(241, 245, 249)        # #F1F5F9
WHITE = RGBColor(255, 255, 255)
GREEN_SUCCESS = RGBColor(22, 163, 74)    # #16A34A


def set_slide_background(slide, color=WHITE):
    """Fill slide background with a solid color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_slide_header(slide, title_text, category="PREDICTIVE ANALYTICS PROJECT"):
    """Add unified header bar across content slides."""
    # Category tag
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.4), Inches(0.3))
    p_cat = tb_cat.text_frame.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = "Segoe UI"
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = BLUE_ACCENT

    # Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(8.4), Inches(0.6))
    p_title = tb_title.text_frame.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Segoe UI"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY_PRIMARY


def create_presentation():
    """Build the 10-slide PowerPoint presentation."""
    prs = Presentation()
    prs.slide_width = Inches(10.0)
    prs.slide_height = Inches(5.625)  # 16:9 Widescreen standard

    blank_layout = prs.slide_layouts[6]

    # =========================================================
    # SLIDE 1: Title Slide
    # =========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, WHITE)

    # Decorative left accent bar
    accent_bar = slide1.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.12), Inches(3.2)
    )
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = BLUE_ACCENT
    accent_bar.line.fill.background()

    # Title textbox
    tb1 = slide1.shapes.add_textbox(Inches(1.1), Inches(1.15), Inches(8.1), Inches(3.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "ACADEMIC FIELD SURVEY & PREDICTIVE ANALYTICS"
    p0.font.name = "Segoe UI"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = BLUE_ACCENT

    p1 = tf1.add_paragraph()
    p1.text = "Predict Whether a Person Will Make an Online Purchase Using Machine Learning"
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = NAVY_PRIMARY
    p1.space_before = Pt(8)
    p1.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "End-to-End Classification System on N=1,200 Verified Shopper Survey Responses"
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(13)
    p2.font.italic = True
    p2.font.color.rgb = TEXT_MUTED
    p2.space_after = Pt(24)

    p3 = tf1.add_paragraph()
    p3.text = "Student: Malatesh D   |   Faculty Evaluator: Prof. Debarshi Mazumder\nCourse: Predictive Analytics (Semester VII)   |   Academic Year: 2026–2027"
    p3.font.name = "Segoe UI"
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = TEXT_DARK

    # =========================================================
    # SLIDE 2: Business Problem & Task Formulation
    # =========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, WHITE)
    add_slide_header(slide2, "Business Problem & Task Formulation")

    # 3 Problem Cards
    col_w = Inches(2.65)
    top_pos = Inches(1.5)
    height_pos = Inches(3.4)

    cards_data = [
        (
            "🚨 The Conversion Crisis",
            "• Average digital retail conversion rate is only 2.0% – 3.5%.\n• Over 96% of website visitors exit without completing an order.\n• Millions of dollars in ad spend are wasted on casual window shoppers.",
        ),
        (
            "🎯 Predictive Task",
            "• Supervised Binary Classification:\n  y ∈ {'Yes': 1, 'No': 0}\n• Predict real-time transaction completion given 12 observable demographic and behavioural attributes.\n• Estimate order expenditure ($).",
        ),
        (
            "💡 Commercial Value",
            "• Targeted Cart Recovery: Alert merchandisers to high-intent shoppers.\n• Margin Protection: Restrict discount popups to nudgeable visitors.\n• Mobile Optimization: Eliminate checkout drop-off friction.",
        ),
    ]

    for i, (head, body) in enumerate(cards_data):
        left_pos = Inches(0.8 + i * 2.9)
        card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, top_pos, col_w, height_pos)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = RGBColor(226, 232, 240)
        card.line.width = Pt(1)

        tb = slide2.shapes.add_textbox(left_pos + Inches(0.15), top_pos + Inches(0.15), col_w - Inches(0.3), height_pos - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.name = "Segoe UI"
        p_h.font.size = Pt(12.5)
        p_h.font.bold = True
        p_h.font.color.rgb = NAVY_PRIMARY
        p_h.space_after = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Segoe UI"
        p_b.font.size = Pt(9.5)
        p_b.font.color.rgb = TEXT_DARK

    # =========================================================
    # SLIDE 3: Academic Survey Dataset (N=1,200)
    # =========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, WHITE)
    add_slide_header(slide3, "Academic Field Survey Architecture (N=1,200)")

    # Left KPI Summary
    tb_kpi = slide3.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(3.8), Inches(3.5))
    tf_kpi = tb_kpi.text_frame
    tf_kpi.word_wrap = True

    p = tf_kpi.paragraphs[0]
    p.text = "Survey Methodology & Structure"
    p.font.name = "Segoe UI"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY
    p.space_after = Pt(8)

    p_body = tf_kpi.add_paragraph()
    p_body.text = (
        "• Sample Population: N = 1,200 verified shoppers.\n"
        "• Questionnaire: 14 administered items capturing demographics, session duration, and cart items.\n"
        "• Natural Class Imbalance:\n"
        "  - Completed Purchases: 349 (29.1%)\n"
        "  - Non-Purchasers: 851 (70.9%)\n"
        "  - Imbalance Ratio: 2.44 : 1\n"
        "• Deliverables: Both raw CSV and formatted Excel (.xlsx with Codebook) generated."
    )
    p_body.font.name = "Segoe UI"
    p_body.font.size = Pt(10)
    p_body.font.color.rgb = TEXT_DARK

    # Right: Embedded Chart (Target Distribution)
    img_target = VISUALIZATIONS_DIR / "target_distribution.png"
    if img_target.exists():
        slide3.shapes.add_picture(str(img_target), Inches(4.8), Inches(1.4), width=Inches(4.4))

    # =========================================================
    # SLIDE 4: Key Findings from EDA
    # =========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, WHITE)
    add_slide_header(slide4, "Exploratory Data Analysis: Key Behavioral Insights")

    # Left: Insights List
    tb_eda = slide4.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(4.2), Inches(3.6))
    tf_eda = tb_eda.text_frame
    tf_eda.word_wrap = True

    insights_points = [
        ("🛒 Cart Page Value Disparity:", "Buyers average $75.82 in cart score vs only $15.65 for non-buyers (4.8x higher commitment)."),
        ("🚪 Exit Rate Velocity:", "Non-buyers display 4.31% drop-off exit rates vs 2.44% for buyers. Exits above 4.0% signal immediate abandonment."),
        ("💻 Hardware Platform Gap:", "Laptops/Desktops achieve 38.4% conversion rate vs 24.8% on Smartphones due to mobile checkout friction."),
        ("📣 Marketing Attribution:", "Direct visits (36.2%) and Google search (33.1%) outperform broad social ads (22.4%)."),
    ]
    for i, (h_text, b_text) in enumerate(insights_points):
        p_h = tf_eda.add_paragraph() if i > 0 else tf_eda.paragraphs[0]
        p_h.text = h_text
        p_h.font.name = "Segoe UI"
        p_h.font.size = Pt(10.5)
        p_h.font.bold = True
        p_h.font.color.rgb = NAVY_PRIMARY

        p_b = tf_eda.add_paragraph()
        p_b.text = b_text
        p_b.font.name = "Segoe UI"
        p_b.font.size = Pt(9.5)
        p_b.font.color.rgb = TEXT_DARK
        p_b.space_after = Pt(6)

    # Right: Embedded Chart (Cart Value Distribution)
    img_cart = VISUALIZATIONS_DIR / "cart_value_distribution.png"
    if img_cart.exists():
        slide4.shapes.add_picture(str(img_cart), Inches(5.1), Inches(1.4), width=Inches(4.1))

    # =========================================================
    # SLIDE 5: Machine Learning Pipeline & Leakage Prevention
    # =========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, WHITE)
    add_slide_header(slide5, "Feature Engineering & Preprocessing Pipeline")

    tb_pipe = slide5.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(8.4), Inches(3.6))
    tf_pipe = tb_pipe.text_frame
    tf_pipe.word_wrap = True

    p = tf_pipe.paragraphs[0]
    p.text = "Leakage-Proof Pipeline Architecture (ColumnTransformer)"
    p.font.name = "Segoe UI"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY
    p.space_after = Pt(6)

    p_pipe_body = tf_pipe.add_paragraph()
    p_pipe_body.text = (
        "1. Strict Train-Test Partitioning:\n"
        "   • Stratified 80:20 split: N_train = 960 records, N_test = 240 records.\n"
        "   • Preserves exact 29.1% positive ratio across splits; prevents data contamination.\n\n"
        "2. Numerical Transformations (6 Features):\n"
        "   • Median Imputation + StandardScaler: Zero-centers mean and scales variance to unity.\n"
        "   • Features: Age, Products Viewed, Browsing Time, Cart Pages, Cart Score, Exit Rate.\n\n"
        "3. Categorical Transformations (6 Features -> 26 One-Hot Columns):\n"
        "   • Most-Frequent Imputation + OneHotEncoder(handle_unknown='ignore', sparse_output=False).\n"
        "   • Resulting Design Matrix: 32 dense, standardized features ready for parametric inference.\n\n"
        "4. Threshold Optimization:\n"
        "   • Optimized decision boundary (tau = 0.83) via grid search to maximize F1-score on the buyer class."
    )
    p_pipe_body.font.name = "Segoe UI"
    p_pipe_body.font.size = Pt(9.5)
    p_pipe_body.font.color.rgb = TEXT_DARK

    # =========================================================
    # SLIDE 6: Multi-Model Benchmark (Champion Selection)
    # =========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, WHITE)
    add_slide_header(slide6, "Model Benchmarking & Champion Selection")

    # Comparison Table
    rows = 6
    cols = 7
    tbl_shape = slide6.shapes.add_table(rows, cols, Inches(0.8), Inches(1.4), Inches(8.4), Inches(2.2))
    table = tbl_shape.table

    table_headers = ["Model", "Accuracy", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "Tuned F1"]
    for j, h in enumerate(table_headers):
        cell = table.cell(0, j)
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Segoe UI"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_PRIMARY

    models_data = [
        ("Logistic Regression 🏆", "86.25%", "90.00%", "0.7925", "0.9602", "0.9225", "0.8372"),
        ("Gradient Boosting", "88.75%", "85.71%", "0.8163", "0.9532", "0.8980", "0.8267"),
        ("XGBoost Classifier", "85.83%", "85.71%", "0.7792", "0.9513", "0.9033", "0.8189"),
        ("Random Forest", "81.25%", "88.57%", "0.7337", "0.9349", "0.8466", "0.8112"),
        ("Decision Tree", "82.08%", "92.86%", "0.7514", "0.9245", "0.8742", "0.7704"),
    ]

    for i, r_data in enumerate(models_data, start=1):
        for j, val in enumerate(r_data):
            cell = table.cell(i, j)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Segoe UI"
            p.font.size = Pt(8.5)
            if i == 1:
                p.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(239, 246, 255)
            elif i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = CARD_BG

    # Bottom takeaway box
    tb_takeaway = slide6.shapes.add_textbox(Inches(0.8), Inches(3.8), Inches(8.4), Inches(1.2))
    tf_takeaway = tb_takeaway.text_frame
    tf_takeaway.word_wrap = True

    p_t = tf_takeaway.paragraphs[0]
    p_t.text = "🏆 Why Logistic Regression was Selected as Champion:"
    p_t.font.name = "Segoe UI"
    p_t.font.size = Pt(11)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY_PRIMARY

    p_t_b = tf_takeaway.add_paragraph()
    p_t_b.text = (
        "• Highest Discriminative Power: 0.9602 ROC-AUC and 0.9225 PR-AUC (exceeding baseline by 3.1x).\n"
        "• Outstanding Buyer Recall: Captures 90.00% of purchasers (63 of 70 holdout buyers).\n"
        "• Full Mathematical Transparency: Directly interpretable odds ratios without ensemble black-box risks."
    )
    p_t_b.font.name = "Segoe UI"
    p_t_b.font.size = Pt(9.5)
    p_t_b.font.color.rgb = TEXT_DARK

    # =========================================================
    # SLIDE 7: Feature Importance & Interpretability
    # =========================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, WHITE)
    add_slide_header(slide7, "Model Interpretability: Key Drivers of Purchase")

    # Left: Explanation
    tb_feat = slide7.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(4.2), Inches(3.6))
    tf_feat = tb_feat.text_frame
    tf_feat.word_wrap = True

    p = tf_feat.paragraphs[0]
    p.text = "Feature Weights & Odds Multipliers"
    p.font.name = "Segoe UI"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY
    p.space_after = Pt(6)

    p_feat_b = tf_feat.add_paragraph()
    p_feat_b.text = (
        "1. Cart_Page_Value_Score (+1.482 weight, 4.40x odds):\n"
        "   • Decisive positive driver. Every 1 SD increase quadruples conversion odds.\n\n"
        "2. Drop_Off_Exit_Rate_Pct (-1.115 weight, 0.33x odds):\n"
        "   • Strongest negative inhibitor. High exit rates reduce purchase likelihood by 67%.\n\n"
        "3. Cart_Checkout_Pages (+0.642 weight, 1.90x odds):\n"
        "   • Reaching shipping/billing pages doubles completion odds.\n\n"
        "4. Primary_Device: Laptop (+0.365 weight, 1.44x odds):\n"
        "   • Desktop navigation provides measurable lift over mobile browsing."
    )
    p_feat_b.font.name = "Segoe UI"
    p_feat_b.font.size = Pt(9.5)
    p_feat_b.font.color.rgb = TEXT_DARK

    # Right: Embedded Chart (Feature Importance)
    img_feat = VISUALIZATIONS_DIR / "feature_importance.png"
    if img_feat.exists():
        slide7.shapes.add_picture(str(img_feat), Inches(5.1), Inches(1.4), width=Inches(4.1))

    # =========================================================
    # SLIDE 8: Streamlit Application Architecture
    # =========================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, WHITE)
    add_slide_header(slide8, "Interactive Streamlit Web Dashboard")

    col_w8 = Inches(2.65)
    top_pos8 = Inches(1.5)
    height_pos8 = Inches(3.4)

    app_cards = [
        (
            "🏠 1. Home Dashboard",
            "• Executive KPIs: 1,200 respondents, 29.1% conversion rate, 5 models benchmarked.\n• Device & Referral distribution charts.\n• Survey background and download buttons for CSV and Excel (.xlsx).",
        ),
        (
            "📊 2. Data Explorer",
            "• Multi-column interactive filtering by device, customer type, and channel.\n• Full survey data inspection table.\n• Correlation heatmaps and demographic histograms.",
        ),
        (
            "🔮 3. Purchase Prediction",
            "• Real-time inference form with all 12 survey inputs.\n• Quick presets: Mobile Buyer, Window Shopper, Holiday Shopper.\n• Purchase probability gauge, confidence score, and estimated order value ($).",
        ),
    ]

    for i, (head, body) in enumerate(app_cards):
        left_pos = Inches(0.8 + i * 2.9)
        card = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, top_pos8, col_w8, height_pos8)
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = BLUE_ACCENT
        card.line.width = Pt(1.5)

        tb = slide8.shapes.add_textbox(left_pos + Inches(0.15), top_pos8 + Inches(0.15), col_w8 - Inches(0.3), height_pos8 - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.name = "Segoe UI"
        p_h.font.size = Pt(12)
        p_h.font.bold = True
        p_h.font.color.rgb = NAVY_PRIMARY
        p_h.space_after = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Segoe UI"
        p_b.font.size = Pt(9.5)
        p_b.font.color.rgb = TEXT_DARK

    # =========================================================
    # SLIDE 9: Strategic Business Recommendations
    # =========================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, WHITE)
    add_slide_header(slide9, "Actionable Business Recommendations")

    tb_recs = slide9.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.6))
    tf_recs = tb_recs.text_frame
    tf_recs.word_wrap = True

    recs_list = [
        ("1. Dynamic Exit-Intent 'Cart Rescue':", "When an active session exhibits Cart_Page_Value > $40 but Exit_Rate approaches 3.5%, trigger an automated overlay offering free expedited shipping or 1-click checkout."),
        ("2. Frictionless Mobile Checkout Overhaul:", "Close the 13.6% conversion gap between Smartphones (24.8%) and Laptops (38.4%) by implementing Apple Pay, Google Pay, and single-tap guest checkout."),
        ("3. Reallocate Acquisition Marketing Budget:", "Shift 20%–30% of top-of-funnel social media advertising budget into high-intent Google Search and targeted lifecycle email campaigns, which yield 1.5x higher conversion."),
        ("4. Fast-Track High-Intent Shoppers (>80% Probability):", "For visitors classified as High-Intent, suppress all secondary survey prompts, newsletter banners, and popups to ensure an unhindered checkout path."),
    ]

    for i, (h_text, b_text) in enumerate(recs_list):
        p_h = tf_recs.add_paragraph() if i > 0 else tf_recs.paragraphs[0]
        p_h.text = h_text
        p_h.font.name = "Segoe UI"
        p_h.font.size = Pt(11)
        p_h.font.bold = True
        p_h.font.color.rgb = NAVY_PRIMARY

        p_b = tf_recs.add_paragraph()
        p_b.text = b_text
        p_b.font.name = "Segoe UI"
        p_b.font.size = Pt(9.5)
        p_b.font.color.rgb = TEXT_DARK
        p_b.space_after = Pt(8)

    # =========================================================
    # SLIDE 10: Conclusion & Project Links
    # =========================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, WHITE)
    add_slide_header(slide10, "Summary & Academic Submission Details")

    tb_end = slide10.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(8.4), Inches(3.5))
    tf_end = tb_end.text_frame
    tf_end.word_wrap = True

    p = tf_end.paragraphs[0]
    p.text = "Project Summary & Verification Checklist:"
    p.font.name = "Segoe UI"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY
    p.space_after = Pt(8)

    p_end_b = tf_end.add_paragraph()
    p_end_b.text = (
        "✅ Academic Field Survey: 1,200 manual responses in CSV and styled Excel workbook.\n"
        "✅ Machine Learning Performance: Logistic Regression achieves 86.25% Accuracy & 96.02% ROC-AUC.\n"
        "✅ Streamlit Frontend: Live interactive web application with pure white theme (localhost:8501).\n"
        "✅ Complete Academic Documentation: Detailed project report (.docx, .md, .html) matching rubric.\n"
        "✅ GitHub Repository: Fully committed and pushed to remote branch.\n\n"
        "🔗 GitHub Repo: https://github.com/Malatesh-D/Online-Purchase-Prediction.git\n"
        "🎓 Course: Predictive Analytics (Semester VII)   |   Faculty: Prof. Debarshi Mazumder\n"
        "👤 Submitted By: Malatesh D"
    )
    p_end_b.font.name = "Segoe UI"
    p_end_b.font.size = Pt(10.5)
    p_end_b.font.color.rgb = TEXT_DARK

    # Save presentation
    output_path = REPORTS_DIR / "PROJECT_PRESENTATION.pptx"
    prs.save(str(output_path))
    print(f"Presentation successfully saved to {output_path}")
    return output_path


if __name__ == "__main__":
    create_presentation()

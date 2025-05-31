def get_workflow_css():
    return """
    <style>
    .lane {
        border: 1px solid #ddd;
        border-radius: 10px;
        margin: 1.5em 0;
        background-color: #ffffff;
        font-family: sans-serif;
    }
    .header {
        padding: 10px;
        font-weight: bold;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        color: white;
    }
    .model { background-color: #0d6efd; }
    .stress { background-color: #198754; }
    .rationale { background-color: #6f42c1; }

    .horizontal-body {
        padding: 1em;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .step-model, .step-stress, .step-rationale {
        padding: 10px 14px;
        background: #e7f1ff;
        border-radius: 8px;
        font-size: 0.9em;
        color: #084298;
        font-weight: 500;
        opacity: 0;
        animation: fadeIn 0.8s ease-in forwards;
    }

    .completed {
        border: 2px solid #084298;
        background-color: #d0e2ff;
    }

    .arrow-model, .arrow-stress, .arrow-rationale {
        font-size: 1em;
        line-height: 1;
        transform: scaleX(1.2);
    }
    .arrow-model { color: #0d6efd; }
    .arrow-stress { color: #198754; }
    .arrow-rationale { color: #6f42c1; }

    @keyframes fadeIn {
        to {
            opacity: 1;
        }
    }
    </style>
    """

def get_workflow_html_ml(step):
    def box(text, index):
        class_name = "step-model completed" if step > index else "step-model"
        delay = f"{0.2 * index:.1f}s"
        return f'<div class="{class_name}" style="animation-delay: {delay};">{text}</div>'

    html = f"""
    <div class="lane">
        <div class="header model">Model Inference</div>
        <div class="horizontal-body">
            {box("Prepare ML Input (JSON)", 0)}
            <div class="arrow-model">▶</div>
            {box("Featurization", 1)}
            <div class="arrow-model">▶</div>
            {box("ML Model Inference", 2)}
        </div>
    </div>
    """
    return html

def get_workflow_html_rf(step):
    def box(text, index):
        class_name = "step-stress completed" if step > index else "step-stress"
        delay = f"{0.2 * index:.1f}s"
        return f'<div class="{class_name}" style="animation-delay: {delay};">{text}</div>'

    html = f"""
    <div class="lane">
        <div class="header stress">Risk Factor Inference</div>
        <div class="horizontal-body">
            {box("Source Risk Factors", 0)}
            <div class="arrow-stress">▶</div>
            {box("Run IR Delta Observability Test", 1)}
            <div class="arrow-stress">▶</div>
            {box("Run Volatality Observability Test", 2)}
            <div class="arrow-stress">▶</div>
            {box("Assess observability of Total PV", 3)}
        </div>
    </div>
    """
    return html

def get_workflow_html_rat(step):
    def box(text, index):
        class_name = "step-rationale completed" if step > index else "step-rationale"
        delay = f"{0.2 * index:.1f}s"
        return f'<div class="{class_name}" style="animation-delay: {delay};">{text}</div>'

    html = f"""
    <div class="lane">
        <div class="header rationale">Rationale Explanation</div>
        <div class="horizontal-body">
            {box("Collect Evidence", 0)}
            <div class="arrow-rationale">▶</div>
            {box("Query GPT", 1)}
            <div class="arrow-rationale">▶</div>
            {box("Summarize Insight", 2)}
        </div>
    </div>
    """
    return html

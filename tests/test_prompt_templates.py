from scripts.run_eval import PROMPT_TEMPLATES

def test_support_ticket_prompt_template_renders():
    tmpl = PROMPT_TEMPLATES.get("support_ticket_classification")
    assert tmpl is not None

    sample = tmpl.format(text="The app crashes on startup.")
    assert "The app crashes on startup." in sample
    assert "billing" in sample
    assert "technical_issue" in sample

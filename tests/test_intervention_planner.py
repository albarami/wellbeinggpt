"""Tests for World Model intervention planner.

Tests:
- Framework-bound step validation
- No medical claims gate
- Leading indicators handling
- Risk detection from negative edges
"""

import pytest

from apps.api.core.world_model.schemas import (
    validate_no_medical_claims,
    InterventionStep,
    InterventionPlan,
    FORBIDDEN_CLAIMS,
)
from apps.api.core.world_model.intervention_planner import (
    validate_intervention_plan,
    intervention_plan_to_dict,
)


class TestMedicalClaimsGate:
    """Tests for the no-diagnostics / no-medical-claims gate."""
    
    def test_safe_text_passes(self):
        """Normal wellbeing text passes validation."""
        assert validate_no_medical_claims("تحقيق التوازن النفسي")
        assert validate_no_medical_claims("تعزيز الصحة الروحية")
        assert validate_no_medical_claims("التركيز على القيم الإيجابية")
    
    def test_medical_claims_fail(self):
        """Medical/diagnostic claims fail validation."""
        assert not validate_no_medical_claims("يعالج الاكتئاب")
        assert not validate_no_medical_claims("دواء للقلق")
        assert not validate_no_medical_claims("تشخيص الحالة النفسية")
        assert not validate_no_medical_claims("علاج طبي للمرض")
    
    def test_forbidden_claims_list(self):
        """All forbidden claims are detected."""
        for claim in FORBIDDEN_CLAIMS:
            text = f"هذا النص يحتوي على {claim}"
            assert not validate_no_medical_claims(text), f"Should detect: {claim}"
    
    def test_empty_text_passes(self):
        """Empty text passes validation."""
        assert validate_no_medical_claims("")
        assert validate_no_medical_claims(None)
    
    def test_similar_safe_words_pass(self):
        """Words similar to forbidden ones but safe should pass."""
        # "صحة" (health) is allowed, but not "علاج طبي"
        assert validate_no_medical_claims("الصحة النفسية")
        assert validate_no_medical_claims("صحة البدن")


class TestInterventionPlanValidation:
    """Tests for intervention plan validation."""
    
    def test_valid_plan_passes(self):
        """A valid plan with no issues passes."""
        plan = InterventionPlan(
            goal_ar="تحقيق التوازن",
            steps=[
                InterventionStep(
                    target_node_ref_kind="pillar",
                    target_node_ref_id="P001",
                    target_node_label_ar="الركيزة الروحية",
                    mechanism_reason_ar="تعزيز القيم",
                    mechanism_citations=[],
                    expected_impacts=[],
                    impact_citations=[],
                ),
            ],
            leading_indicators=[],
            risk_of_imbalance=[],
        )
        
        issues = validate_intervention_plan(plan)
        assert len(issues) == 0
    
    def test_medical_claim_in_goal_fails(self):
        """Medical claim in goal is detected."""
        plan = InterventionPlan(
            goal_ar="يعالج الاكتئاب السريري",
            steps=[],
            leading_indicators=[],
            risk_of_imbalance=[],
        )
        
        issues = validate_intervention_plan(plan)
        assert any("Goal contains forbidden medical claim" in i for i in issues)
    
    def test_medical_claim_in_step_fails(self):
        """Medical claim in step is detected."""
        plan = InterventionPlan(
            goal_ar="تحقيق التوازن",
            steps=[
                InterventionStep(
                    target_node_ref_kind="pillar",
                    target_node_ref_id="P001",
                    target_node_label_ar="يشفي المرض",  # Forbidden
                    mechanism_reason_ar="سبب",
                    mechanism_citations=[],
                    expected_impacts=[],
                    impact_citations=[],
                ),
            ],
            leading_indicators=[],
            risk_of_imbalance=[],
        )
        
        issues = validate_intervention_plan(plan)
        assert any("Step 1 target contains forbidden medical claim" in i for i in issues)
    
    def test_missing_node_mapping_fails(self):
        """Step without framework node mapping is detected."""
        plan = InterventionPlan(
            goal_ar="تحقيق التوازن",
            steps=[
                InterventionStep(
                    target_node_ref_kind="",  # Missing
                    target_node_ref_id="",     # Missing
                    target_node_label_ar="خطوة",
                    mechanism_reason_ar="سبب",
                    mechanism_citations=[],
                    expected_impacts=[],
                    impact_citations=[],
                ),
            ],
            leading_indicators=[],
            risk_of_imbalance=[],
        )
        
        issues = validate_intervention_plan(plan)
        assert any("missing framework node mapping" in i for i in issues)


class TestInterventionPlanSerialization:
    """Tests for plan serialization."""
    
    def test_to_dict_structure(self):
        """Plan converts to correct dictionary structure."""
        plan = InterventionPlan(
            goal_ar="الهدف",
            steps=[
                InterventionStep(
                    target_node_ref_kind="pillar",
                    target_node_ref_id="P001",
                    target_node_label_ar="الروحية",
                    mechanism_reason_ar="السبب",
                    mechanism_citations=[{"chunk_id": "c1"}],
                    expected_impacts=["تأثير 1"],
                    impact_citations=[],
                ),
            ],
            leading_indicators=[
                {"indicator_ar": "مؤشر", "source": "framework"},
            ],
            risk_of_imbalance=[
                {"risk_ar": "خطر", "affected_pillar": "P002"},
            ],
        )
        
        d = intervention_plan_to_dict(plan)
        
        assert d["goal_ar"] == "الهدف"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["target_node_ref_kind"] == "pillar"
        assert d["steps"][0]["target_node_ref_id"] == "P001"
        assert len(d["leading_indicators"]) == 1
        assert d["leading_indicators"][0]["source"] == "framework"
        assert len(d["risk_of_imbalance"]) == 1


class TestLeadingIndicators:
    """Tests for leading indicators handling."""
    
    def test_framework_indicator_has_source(self):
        """Indicators from framework have source='framework'."""
        plan = InterventionPlan(
            goal_ar="الهدف",
            steps=[],
            leading_indicators=[
                {"indicator_ar": "مؤشر من الإطار", "source": "framework"},
            ],
            risk_of_imbalance=[],
        )
        
        d = intervention_plan_to_dict(plan)
        assert d["leading_indicators"][0]["source"] == "framework"
    
    def test_missing_indicator_has_placeholder_source(self):
        """Missing indicators have source='غير منصوص'."""
        plan = InterventionPlan(
            goal_ar="الهدف",
            steps=[],
            leading_indicators=[
                {"indicator_ar": "غير محدد", "source": "غير منصوص"},
            ],
            risk_of_imbalance=[],
        )
        
        d = intervention_plan_to_dict(plan)
        assert d["leading_indicators"][0]["source"] == "غير منصوص"


class TestRiskDetection:
    """Tests for risk detection from negative edges."""
    
    def test_risk_has_affected_pillar(self):
        """Risk entries include affected pillar."""
        plan = InterventionPlan(
            goal_ar="الهدف",
            steps=[],
            leading_indicators=[],
            risk_of_imbalance=[
                {
                    "risk_ar": "INHIBITS: الروحية → البدنية",
                    "affected_pillar": "P004",
                    "evidence": [],
                },
            ],
        )
        
        d = intervention_plan_to_dict(plan)
        assert d["risk_of_imbalance"][0]["affected_pillar"] == "P004"
    
    def test_risk_includes_evidence(self):
        """Risk entries can include evidence."""
        plan = InterventionPlan(
            goal_ar="الهدف",
            steps=[],
            leading_indicators=[],
            risk_of_imbalance=[
                {
                    "risk_ar": "تعارض",
                    "affected_pillar": "P002",
                    "evidence": [{"chunk_id": "c1", "quote": "دليل"}],
                },
            ],
        )
        
        d = intervention_plan_to_dict(plan)
        assert len(d["risk_of_imbalance"][0]["evidence"]) == 1

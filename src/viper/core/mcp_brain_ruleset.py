#!/usr/bin/env python3
"""
🧠 VIPER MCP BRAIN RULESET
The Comprehensive Governance System for MCP Operations

This is the MASTER RULESET that governs all MCP brain behavior:
- System operation rules and constraints
- Trading decision frameworks
- AI control parameters
- Security and compliance protocols
- Performance optimization guidelines
- Emergency response procedures

RULESET HIERARCHY:
1. Core System Rules (immutable)
2. Operational Rules (configurable)
3. Trading Rules (dynamic)
4. AI Control Rules (adaptive)
5. Security Rules (enforceable)
6. Emergency Protocols (override)
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum

class RulePriority(Enum):
    """Rule priority levels"""
    CRITICAL = "critical"    # System-breaking, cannot be overridden
    HIGH = "high"           # Important, requires approval to override
    MEDIUM = "medium"       # Standard, can be adjusted
    LOW = "low"            # Flexible, auto-adjustable

class RuleCategory(Enum):
    """Rule categories"""
    SYSTEM = "system"
    TRADING = "trading"
    AI_CONTROL = "ai_control"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EMERGENCY = "emergency"

@dataclass
class MCPRule:
    """Individual MCP rule definition"""
    name: str
    category: RuleCategory
    priority: RulePriority
    description: str
    value: Any
    constraints: Dict[str, Any] = field(default_factory=dict)
    enforcement: str = "strict"
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_by: str = "system"

    def validate(self, new_value: Any) -> bool:
        """Validate if a new value meets rule constraints"""
        if not self.constraints:
            return True

        # Check type constraints
        if "type" in self.constraints:
            if not isinstance(new_value, self.constraints["type"]):
                return False

        # Check range constraints
        if "min" in self.constraints and new_value < self.constraints["min"]:
            return False
        if "max" in self.constraints and new_value > self.constraints["max"]:
            return False

        # Check enum constraints
        if "allowed_values" in self.constraints:
            if new_value not in self.constraints["allowed_values"]:
                return False

        return True

    def can_override(self, requester: str) -> bool:
        """Check if rule can be overridden by requester"""
        if self.priority == RulePriority.CRITICAL:
            return False
        elif self.priority == RulePriority.HIGH:
            return requester in ["admin", "system", "emergency"]
        else:
            return True

class MCPRulesEngine:
    """The Rules Engine that enforces all MCP governance"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, MCPRule] = {}
        self.rule_violations: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.emergency_mode = False

        # Initialize core rules
        self.initialize_core_rules()

    def initialize_core_rules(self):
        """Initialize the immutable core ruleset"""

        # SYSTEM RULES - Critical, cannot be changed
        self.add_rule(MCPRule(
            name="continuous_operation",
            category=RuleCategory.SYSTEM,
            priority=RulePriority.CRITICAL,
            description="System must operate continuously without interruption",
            value=True,
            enforcement="immutable"
        ))

        self.add_rule(MCPRule(
            name="brain_as_master",
            category=RuleCategory.SYSTEM,
            priority=RulePriority.CRITICAL,
            description="MCP Brain is the master controller of all operations",
            value=True,
            enforcement="immutable"
        ))

        self.add_rule(MCPRule(
            name="auto_restart",
            category=RuleCategory.SYSTEM,
            priority=RulePriority.CRITICAL,
            description="System must automatically restart failed components",
            value=True,
            enforcement="immutable"
        ))

        # OPERATIONAL RULES - High priority, configurable
        self.add_rule(MCPRule(
            name="max_downtime",
            category=RuleCategory.SYSTEM,
            priority=RulePriority.HIGH,
            description="Maximum allowed system downtime in seconds",
            value=300,
            constraints={"type": int, "min": 30, "max": 3600}
        ))

        self.add_rule(MCPRule(
            name="health_check_interval",
            category=RuleCategory.SYSTEM,
            priority=RulePriority.HIGH,
            description="Interval between health checks in seconds",
            value=30,
            constraints={"type": int, "min": 5, "max": 300}
        ))

        # TRADING RULES - Dynamic, high priority
        self.add_rule(MCPRule(
            name="max_concurrent_positions",
            category=RuleCategory.TRADING,
            priority=RulePriority.HIGH,
            description="Maximum number of concurrent trading positions",
            value=15,
            constraints={"type": int, "min": 1, "max": 50}
        ))

        self.add_rule(MCPRule(
            name="max_risk_per_trade",
            category=RuleCategory.TRADING,
            priority=RulePriority.HIGH,
            description="Maximum risk per trade as percentage",
            value=0.02,
            constraints={"type": float, "min": 0.001, "max": 0.1}
        ))

        self.add_rule(MCPRule(
            name="daily_loss_limit",
            category=RuleCategory.TRADING,
            priority=RulePriority.HIGH,
            description="Maximum daily loss limit as percentage",
            value=0.03,
            constraints={"type": float, "min": 0.01, "max": 0.2}
        ))

        self.add_rule(MCPRule(
            name="leverage_limit",
            category=RuleCategory.TRADING,
            priority=RulePriority.HIGH,
            description="Maximum leverage allowed",
            value=25,
            constraints={"type": int, "min": 1, "max": 100}
        ))

        # AI CONTROL RULES - Adaptive, medium priority
        self.add_rule(MCPRule(
            name="cursor_integration_enabled",
            category=RuleCategory.AI_CONTROL,
            priority=RulePriority.HIGH,
            description="Enable Cursor AI integration",
            value=True,
            enforcement="strict"
        ))

        self.add_rule(MCPRule(
            name="decision_threshold",
            category=RuleCategory.AI_CONTROL,
            priority=RulePriority.MEDIUM,
            description="Minimum confidence threshold for AI decisions",
            value=85,
            constraints={"type": int, "min": 50, "max": 95}
        ))

        self.add_rule(MCPRule(
            name="auto_execution_enabled",
            category=RuleCategory.AI_CONTROL,
            priority=RulePriority.HIGH,
            description="Allow automatic execution of AI recommendations",
            value=False,
            enforcement="approval_required"
        ))

        self.add_rule(MCPRule(
            name="learning_mode",
            category=RuleCategory.AI_CONTROL,
            priority=RulePriority.MEDIUM,
            description="Enable continuous learning from trading outcomes",
            value=True
        ))

        # SECURITY RULES - Critical priority
        self.add_rule(MCPRule(
            name="api_key_encryption",
            category=RuleCategory.SECURITY,
            priority=RulePriority.CRITICAL,
            description="All API keys must be encrypted",
            value=True,
            enforcement="immutable"
        ))

        self.add_rule(MCPRule(
            name="request_validation",
            category=RuleCategory.SECURITY,
            priority=RulePriority.CRITICAL,
            description="All requests must be validated",
            value=True,
            enforcement="immutable"
        ))

        self.add_rule(MCPRule(
            name="audit_logging",
            category=RuleCategory.SECURITY,
            priority=RulePriority.CRITICAL,
            description="All operations must be audit logged",
            value=True,
            enforcement="immutable"
        ))

        self.add_rule(MCPRule(
            name="emergency_shutdown",
            category=RuleCategory.SECURITY,
            priority=RulePriority.CRITICAL,
            description="Emergency shutdown capability must be available",
            value=True,
            enforcement="immutable"
        ))

        # PERFORMANCE RULES - Medium priority
        self.add_rule(MCPRule(
            name="memory_limit",
            category=RuleCategory.PERFORMANCE,
            priority=RulePriority.MEDIUM,
            description="Maximum memory usage percentage",
            value=80,
            constraints={"type": int, "min": 50, "max": 95}
        ))

        self.add_rule(MCPRule(
            name="cpu_limit",
            category=RuleCategory.PERFORMANCE,
            priority=RulePriority.MEDIUM,
            description="Maximum CPU usage percentage",
            value=70,
            constraints={"type": int, "min": 30, "max": 90}
        ))

        # EMERGENCY PROTOCOLS - Override priority
        self.add_rule(MCPRule(
            name="emergency_stop_enabled",
            category=RuleCategory.EMERGENCY,
            priority=RulePriority.CRITICAL,
            description="Emergency stop must be available",
            value=True,
            enforcement="immutable"
        ))

        self.add_rule(MCPRule(
            name="circuit_breaker_enabled",
            category=RuleCategory.EMERGENCY,
            priority=RulePriority.CRITICAL,
            description="Circuit breaker protection enabled",
            value=True,
            enforcement="immutable"
        ))

        self.logger.info(f"🛡️ Initialized {len(self.rules)} core rules")

    def add_rule(self, rule: MCPRule):
        """Add a new rule to the ruleset"""
        self.rules[rule.name] = rule
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "rule_added",
            "rule_name": rule.name,
            "rule_value": rule.value,
            "modified_by": rule.modified_by
        })
        self.logger.info(f"📝 Rule added: {rule.name} = {rule.value}")

    def update_rule(self, rule_name: str, new_value: Any, requester: str = "system") -> bool:
        """Update a rule value with proper validation and permissions"""
        if rule_name not in self.rules:
            self.logger.error(f"❌ Rule not found: {rule_name}")
            return False

        rule = self.rules[rule_name]

        # Check permissions
        if not rule.can_override(requester):
            self.log_violation(rule_name, f"Insufficient permissions for {requester}", requester)
            return False

        # Validate new value
        if not rule.validate(new_value):
            self.log_violation(rule_name, f"Invalid value: {new_value}", requester)
            return False

        # Update rule
        old_value = rule.value
        rule.value = new_value
        rule.last_modified = datetime.now().isoformat()
        rule.modified_by = requester

        # Log the change
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "rule_updated",
            "rule_name": rule_name,
            "old_value": old_value,
            "new_value": new_value,
            "modified_by": requester
        })

        self.logger.info(f"📝 Rule updated: {rule_name} = {new_value} (by {requester})")
        return True

    def get_rule_value(self, rule_name: str) -> Any:
        """Get the current value of a rule"""
        if rule_name in self.rules:
            return self.rules[rule_name].value
        return None

    def validate_operation(self, operation: str, params: Dict[str, Any], requester: str = "system") -> Dict[str, Any]:
        """Validate an operation against the ruleset"""
        violations = []
        warnings = []

        # Check trading operations
        if operation in ["execute_trade", "start_live_trading"]:
            # Check risk limits
            risk_per_trade = params.get("risk_per_trade", 0.02)
            max_risk = self.get_rule_value("max_risk_per_trade")

            if risk_per_trade > max_risk:
                violations.append(f"Risk per trade ({risk_per_trade}) exceeds limit ({max_risk})")

            # Check position limits
            max_positions = self.get_rule_value("max_concurrent_positions")
            if params.get("max_positions", 15) > max_positions:
                violations.append(f"Position limit exceeds maximum ({max_positions})")

        # Check AI operations
        elif operation in ["cursor_command", "execute_ai_recommendations"]:
            if not self.get_rule_value("cursor_integration_enabled"):
                violations.append("Cursor integration is disabled")

            confidence = params.get("confidence", 0)
            threshold = self.get_rule_value("decision_threshold")

            if confidence < threshold:
                warnings.append(f"Confidence ({confidence}%) below threshold ({threshold}%)")

        # Check emergency operations
        elif operation in ["emergency_stop", "shutdown_system"]:
            if not self.get_rule_value("emergency_stop_enabled"):
                violations.append("Emergency operations are disabled")

        return {
            "approved": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "requires_approval": len(warnings) > 0 or operation in ["emergency_stop", "shutdown_system"]
        }

    def log_violation(self, rule_name: str, violation: str, requester: str):
        """Log a rule violation"""
        violation_record = {
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule_name,
            "violation": violation,
            "requester": requester,
            "rule_value": self.get_rule_value(rule_name)
        }

        self.rule_violations.append(violation_record)
        self.audit_log.append({
            "timestamp": violation_record["timestamp"],
            "action": "rule_violation",
            "rule_name": rule_name,
            "violation": violation,
            "requester": requester
        })

        self.logger.warning(f"🚨 Rule violation: {rule_name} - {violation} (by {requester})")

    def activate_emergency_mode(self, reason: str, requester: str):
        """Activate emergency mode with override protocols"""
        if not self.get_rule_value("emergency_stop_enabled"):
            self.logger.error("❌ Cannot activate emergency mode - emergency protocols disabled")
            return False

        self.emergency_mode = True

        # Override certain rules in emergency mode
        emergency_overrides = {
            "auto_execution_enabled": False,
            "decision_threshold": 95,
            "max_risk_per_trade": 0.005  # Reduce risk in emergency
        }

        for rule_name, new_value in emergency_overrides.items():
            self.update_rule(rule_name, new_value, "emergency_system")

        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "emergency_mode_activated",
            "reason": reason,
            "requester": requester,
            "overrides": emergency_overrides
        })

        self.logger.critical(f"🚨 EMERGENCY MODE ACTIVATED: {reason} (by {requester})")
        return True

    def deactivate_emergency_mode(self, requester: str):
        """Deactivate emergency mode and restore normal operations"""
        if not self.emergency_mode:
            return False

        self.emergency_mode = False

        # Restore normal rule values
        normal_values = {
            "auto_execution_enabled": False,  # Keep auto execution disabled for safety
            "decision_threshold": 85,
            "max_risk_per_trade": 0.02
        }

        for rule_name, new_value in normal_values.items():
            self.update_rule(rule_name, new_value, "emergency_system")

        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "emergency_mode_deactivated",
            "requester": requester
        })

        self.logger.info(f"✅ EMERGENCY MODE DEACTIVATED (by {requester})")
        return True

    def get_rules_summary(self) -> Dict[str, Any]:
        """Get a summary of all rules and their status"""
        summary = {
            "total_rules": len(self.rules),
            "rules_by_category": {},
            "rules_by_priority": {},
            "recent_violations": len([v for v in self.rule_violations[-10:]]),
            "emergency_mode": self.emergency_mode,
            "last_audit_entries": len(self.audit_log[-5:])
        }

        # Categorize rules
        for rule in self.rules.values():
            summary["rules_by_category"][rule.category.value] = summary["rules_by_category"].get(rule.category.value, 0) + 1
            summary["rules_by_priority"][rule.priority.value] = summary["rules_by_priority"].get(rule.priority.value, 0) + 1

        return summary

    def export_ruleset(self) -> str:
        """Export the complete ruleset as JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "rules": {name: {
                "name": rule.name,
                "category": rule.category.value,
                "priority": rule.priority.value,
                "description": rule.description,
                "value": rule.value,
                "constraints": rule.constraints,
                "enforcement": rule.enforcement,
                "last_modified": rule.last_modified,
                "modified_by": rule.modified_by
            } for name, rule in self.rules.items()},
            "violations": self.rule_violations[-50:],  # Last 50 violations
            "audit_log": self.audit_log[-100:],  # Last 100 audit entries
            "emergency_mode": self.emergency_mode
        }

        return json.dumps(export_data, indent=2, default=str)

    def import_ruleset(self, ruleset_json: str, requester: str = "system") -> bool:
        """Import a ruleset from JSON (admin only)"""
        try:
            import_data = json.loads(ruleset_json)

            # Only allow import by admin
            if requester not in ["admin", "system"]:
                self.logger.error(f"❌ Ruleset import denied for {requester}")
                return False

            # Validate and import rules
            for rule_name, rule_data in import_data.get("rules", {}).items():
                # Create rule object
                rule = MCPRule(
                    name=rule_data["name"],
                    category=RuleCategory(rule_data["category"]),
                    priority=RulePriority(rule_data["priority"]),
                    description=rule_data["description"],
                    value=rule_data["value"],
                    constraints=rule_data.get("constraints", {}),
                    enforcement=rule_data.get("enforcement", "strict"),
                    modified_by=requester
                )

                # Add or update rule
                self.rules[rule_name] = rule

            self.logger.info(f"📥 Ruleset imported successfully by {requester}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ruleset import failed: {e}")
            return False

def main():
    """Test the rules engine"""
    rules_engine = MCPRulesEngine()

    # Test rule validation

    # Test rule update
    success = rules_engine.update_rule("decision_threshold", 90, "admin")

    # Test operation validation
    result = rules_engine.validate_operation("execute_trade", {"risk_per_trade": 0.05})

    # Export ruleset
    ruleset_json = rules_engine.export_ruleset()
    print(f"Ruleset exported: {len(ruleset_json)} characters")


if __name__ == "__main__":
    main()

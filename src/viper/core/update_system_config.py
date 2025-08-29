#!/usr/bin/env python3
"""
# Rocket SYSTEM CONFIGURATION UPDATER
Update system configuration with optimized parameters

This script:
- Loads optimized parameters from parameter optimizer
- Updates the enhanced system configuration
- Validates parameter compatibility
- Provides rollback capabilities
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemConfigUpdater:
    """System configuration updater"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / "enhanced_system_config.json"
        self.backup_path = self.config_path.with_suffix('.backup')

    def load_current_config(self) -> Dict[str, Any]:
        """Load current system configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.error(f"# X Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"# X Error loading config: {e}")
            return {}

    def load_optimized_parameters(self, params_path: Optional[str] = None) -> Dict[str, Any]:
        """Load optimized parameters"""
        try:
            if params_path:
                params_file = Path(params_path)
            else:
                params_file = Path(__file__).parent / "optimized_parameters.json"

            if params_file.exists():
                with open(params_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"# Warning Optimized parameters file not found: {params_file}")
                return {}
        except Exception as e:
            logger.error(f"# X Error loading optimized parameters: {e}")
            return {}

    def create_backup(self) -> bool:
        """Create backup of current configuration"""
        try:
            if self.config_path.exists():
                import shutil
                backup_name = f"{self.config_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self.config_path.suffix}"
                backup_path = self.config_path.parent / backup_name

                shutil.copy2(self.config_path, backup_path)
                logger.info(f"💾 Configuration backup created: {backup_path}")
                return True
            else:
                logger.warning("# Warning No configuration file to backup")
                return False
        except Exception as e:
            logger.error(f"# X Error creating backup: {e}")
            return False

    def update_trading_parameters(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Update trading parameters in configuration"""
        try:
            if 'trading_parameters' not in config:
                config['trading_parameters'] = {}

            # Map optimized parameters to config parameters
            parameter_mapping = {
                'risk_per_trade': 'default_risk_per_trade',
                'max_positions': 'max_positions_total',
                'stop_loss_pct': 'stop_loss_pct',
                'take_profit_pct': 'take_profit_pct',
                'trailing_stop_pct': 'trailing_stop_pct',
                'max_leverage': 'max_leverage',
                'max_daily_loss': 'max_daily_loss',
                'max_drawdown': 'max_drawdown',
                'min_viper_score': 'min_viper_score',
                'scan_interval': 'scan_interval',
                'max_trades_per_hour': 'max_trades_per_hour',
                'min_volume_threshold': 'min_volume_threshold',
                'max_spread_threshold': 'max_spread_threshold',
                'pairs_batch_size': 'pairs_batch_size'
            }

            for opt_param, config_param in parameter_mapping.items():
                if opt_param in params:
                    config['trading_parameters'][config_param] = params[opt_param]
                    logger.info(f"# Chart Updated {config_param}: {params[opt_param]}")

            return config

        except Exception as e:
            logger.error(f"# X Error updating trading parameters: {e}")
            return config

    def update_risk_parameters(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk management parameters"""
        try:
            if 'modules' not in config:
                config['modules'] = {}
            if 'enhanced_risk_manager' not in config['modules']:
                config['modules']['enhanced_risk_manager'] = {'config': {}}

            risk_config = config['modules']['enhanced_risk_manager']['config']

            # Update risk parameters
            risk_mappings = {
                'risk_per_trade': 'max_single_position_risk',
                'max_positions': 'max_positions',
                'max_daily_loss': 'max_daily_loss',
                'max_drawdown': 'max_drawdown'
            }

            for opt_param, config_param in risk_mappings.items():
                if opt_param in params:
                    risk_config[config_param] = params[opt_param]

            return config

        except Exception as e:
            logger.error(f"# X Error updating risk parameters: {e}")
            return config

    def update_technical_parameters(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Update technical analysis parameters"""
        try:
            if 'modules' not in config:
                config['modules'] = {}
            if 'enhanced_technical_optimizer' not in config['modules']:
                config['modules']['enhanced_technical_optimizer'] = {'config': {}}

            tech_config = config['modules']['enhanced_technical_optimizer']['config']

            # Update technical parameters
            tech_mappings = {
                'fast_ma_length': 'fast_ma_length',
                'slow_ma_length': 'medium_ma_length',  # Map slow to medium
                'trend_ma_length': 'trend_ma_length',
                'rsi_oversold': 'rsi_oversold_threshold',
                'rsi_overbought': 'rsi_overbought_threshold',
                'macd_signal_threshold': 'macd_signal_threshold',
                'bb_std_dev': 'bb_std_dev',
                'atr_multiplier': 'atr_multiplier_base'
            }

            for opt_param, config_param in tech_mappings.items():
                if opt_param in params:
                    tech_config[config_param] = params[opt_param]

            return config

        except Exception as e:
            logger.error(f"# X Error updating technical parameters: {e}")
            return config

    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate updated configuration"""
        try:
            issues = []

            # Check trading parameters
            tp = config.get('trading_parameters', {})
            if tp.get('default_risk_per_trade', 0) > 0.05:
                issues.append("# Warning Risk per trade is very high (>5%)")
            if tp.get('max_positions_total', 0) > 50:
                issues.append("# Warning Max positions is very high (>50)")

            # Check risk parameters
            risk_config = config.get('modules', {}).get('enhanced_risk_manager', {}).get('config', {})
            if risk_config.get('max_single_position_risk', 0) > 0.1:
                issues.append("# Warning Single position risk is very high (>10%)")

            # Check technical parameters
            tech_config = config.get('modules', {}).get('enhanced_technical_optimizer', {}).get('config', {})
            if tech_config.get('fast_ma_length', 0) >= tech_config.get('medium_ma_length', 0):
                issues.append("# Warning Fast MA length should be less than medium MA length")

            return issues

        except Exception as e:
            logger.error(f"# X Error validating configuration: {e}")
            return [f"Validation error: {e}"]

    def save_configuration(self, config: Dict[str, Any]) -> bool:
        """Save updated configuration"""
        try:
            # Add update metadata
            config['last_updated'] = datetime.now().isoformat()
            config['updated_by'] = 'parameter_optimizer'

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)

            logger.info(f"💾 Configuration updated successfully: {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"# X Error saving configuration: {e}")
            return False

    def update_from_optimized_parameters(self, params_path: Optional[str] = None,
                                       create_backup: bool = True) -> bool:
        """Update system configuration with optimized parameters"""
        try:
            logger.info("# Rocket Updating system configuration with optimized parameters")

            # Load current configuration
            config = self.load_current_config()
            if not config:
                return False

            # Create backup
            if create_backup:
                if not self.create_backup():
                    logger.warning("# Warning Failed to create backup, continuing anyway")

            # Load optimized parameters
            optimized_params = self.load_optimized_parameters(params_path)
            if not optimized_params:
                logger.warning("# Warning No optimized parameters found, using defaults")

            # Update different parameter sections
            config = self.update_trading_parameters(config, optimized_params)
            config = self.update_risk_parameters(config, optimized_params)
            config = self.update_technical_parameters(config, optimized_params)

            # Validate configuration
            validation_issues = self.validate_configuration(config)
            if validation_issues:
                logger.warning("# Warning Configuration validation issues:")
                for issue in validation_issues:
                    logger.warning(f"   {issue}")

            # Save updated configuration
            if self.save_configuration(config):
                logger.info("# Check System configuration updated successfully!")
                return True
            else:
                logger.error("# X Failed to save updated configuration")
                return False

        except Exception as e:
            logger.error(f"# X Error updating configuration: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Update system configuration with optimized parameters')
    parser.add_argument('--params-path', help='Path to optimized parameters JSON file')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
    parser.add_argument('--config-path', help='Path to system configuration file')

    args = parser.parse_args()


    updater = SystemConfigUpdater(args.config_path)

    success = updater.update_from_optimized_parameters(
        params_path=args.params_path,
        create_backup=not args.no_backup
    )

    if success:
        print("# Check Configuration update completed successfully!")
        print("🔄 Restart your trading system to apply the new parameters")
    else:
        print("# Tool Check the logs for detailed error information")

if __name__ == "__main__":
    main()

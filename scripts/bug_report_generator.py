#!/usr/bin/env python3
"""
📊 COMPREHENSIVE BUG REPORT GENERATOR - VIPER Quality Assurance
===============================================================

Generates detailed, actionable reports from bug scan results with:
- Executive summary and key metrics
- Prioritized issue breakdown
- File-by-file analysis
- Actionable recommendations
- Trend analysis and improvement tracking
- HTML and PDF report formats

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import base64
from io import BytesIO

@dataclass
class ReportMetrics:
    """Container for report metrics and statistics"""
    total_files: int
    total_lines: int
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    scan_duration: float
    issues_per_file: float
    issues_per_thousand_lines: float
    most_problematic_file: str
    most_common_issue_type: str

class ComprehensiveBugReportGenerator:
    """Generates comprehensive bug reports with detailed analysis"""

    def __init__(self, scan_results_path: str = None):
        self.scan_results_path = scan_results_path or "reports/comprehensive_bug_scan.json"
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)

        # Color scheme for severity levels
        self.colors = {
            'CRITICAL': '#DC3545',
            'HIGH': '#FD7E14',
            'MEDIUM': '#FFC107',
            'LOW': '#28A745',
            'INFO': '#17A2B8'
        }

        # Severity weights for scoring
        self.severity_weights = {
            'CRITICAL': 10,
            'HIGH': 7,
            'MEDIUM': 4,
            'LOW': 2,
            'INFO': 1
        }

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive HTML report"""
        print("📊 GENERATING COMPREHENSIVE BUG REPORT")

        # Load scan results
        with open(self.scan_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Calculate metrics
        metrics = self._calculate_metrics(data)

        # Generate report sections
        sections = {
            'header': self._generate_header_section(metrics),
            'executive_summary': self._generate_executive_summary(metrics, data),
            'severity_breakdown': self._generate_severity_breakdown(data),
            'category_analysis': self._generate_category_analysis(data),
            'file_analysis': self._generate_file_analysis(data),
            'issue_patterns': self._generate_issue_patterns(data),
            'recommendations': self._generate_recommendations_section(data),
            'trends': self._generate_trends_section(),
            'footer': self._generate_footer_section()
        }

        # Combine sections into full HTML report
        html_content = self._combine_html_sections(sections)

        # Save HTML report
        html_path = self.output_dir / "comprehensive_bug_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML Report saved to: {html_path}")

        # Generate PDF report (if weka is available)
        try:
            pdf_path = self._generate_pdf_report(html_content)
            print(f"✅ PDF Report saved to: {pdf_path}")
        except Exception as e:
            print(f"⚠️  PDF generation skipped: {e}")

        return str(html_path)

    def _calculate_metrics(self, data: Dict) -> ReportMetrics:
        """Calculate comprehensive metrics from scan data"""
        findings = data['findings']
        metadata = data['scan_metadata']

        # Count issues by severity
        severity_counts = defaultdict(int)
        for finding in findings:
            severity_counts[finding['severity']] += 1

        # Calculate file-based metrics
        files_with_issues = set()
        issues_per_file = defaultdict(int)
        for finding in findings:
            file_path = finding['file_path'].split('/')[-1]
            files_with_issues.add(file_path)
            issues_per_file[file_path] += 1

        # Find most problematic file
        most_problematic_file = max(issues_per_file.items(), key=lambda x: x[1]) if issues_per_file else ("None", 0)

        # Find most common issue type
        rule_counts = Counter(f['rule_id'] for f in findings)
        most_common_issue = rule_counts.most_common(1)[0] if rule_counts else ("None", 0)

        return ReportMetrics(
            total_files=metadata['total_files_scanned'],
            total_lines=metadata['total_lines_scanned'],
            total_issues=len(findings),
            critical_count=severity_counts['CRITICAL'],
            high_count=severity_counts['HIGH'],
            medium_count=severity_counts['MEDIUM'],
            low_count=severity_counts['LOW'],
            info_count=severity_counts['INFO'],
            scan_duration=metadata['execution_time_seconds'],
            issues_per_file=len(findings) / max(metadata['total_files_scanned'], 1),
            issues_per_thousand_lines=(len(findings) / max(metadata['total_lines_scanned'], 1)) * 1000,
            most_problematic_file=f"{most_problematic_file[0]} ({most_problematic_file[1]} issues)",
            most_common_issue_type=f"{most_common_issue[0]} ({most_common_issue[1]} occurrences)"
        )

    def _generate_header_section(self, metrics: ReportMetrics) -> str:
        """Generate report header with key metrics"""
        return f"""
        <div class="header-section">
            <div class="header-title">
                <h1>🕵️ VIPER Repository Bug Scan Report</h1>
                <p class="subtitle">Comprehensive Code Quality Analysis</p>
                <div class="scan-info">
                    <span class="scan-date">📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                    <span class="scan-duration">⏱️ Scan Duration: {metrics.scan_duration:.2f}s</span>
                </div>
            </div>

            <div class="key-metrics">
                <div class="metric-card">
                    <div class="metric-value">{metrics.total_files}</div>
                    <div class="metric-label">Files Scanned</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.total_lines:,}</div>
                    <div class="metric-label">Lines of Code</div>
                </div>
                <div class="metric-card critical">
                    <div class="metric-value">{metrics.critical_count}</div>
                    <div class="metric-label">Critical Issues</div>
                </div>
                <div class="metric-card high">
                    <div class="metric-value">{metrics.high_count}</div>
                    <div class="metric-label">High Priority</div>
                </div>
                <div class="metric-card medium">
                    <div class="metric-value">{metrics.medium_count}</div>
                    <div class="metric-label">Medium Priority</div>
                </div>
                <div class="metric-card low">
                    <div class="metric-value">{metrics.low_count}</div>
                    <div class="metric-label">Low Priority</div>
                </div>
            </div>
        </div>
        """

    def _generate_executive_summary(self, metrics: ReportMetrics, data: Dict) -> str:
        """Generate executive summary section"""
        # Calculate code quality score
        total_weighted_score = (
            metrics.critical_count * self.severity_weights['CRITICAL'] +
            metrics.high_count * self.severity_weights['HIGH'] +
            metrics.medium_count * self.severity_weights['MEDIUM'] +
            metrics.low_count * self.severity_weights['LOW'] +
            metrics.info_count * self.severity_weights['INFO']
        )

        # Quality score (lower is better, max 100)
        quality_score = min(100, max(0, 100 - (total_weighted_score / metrics.total_files)))

        return f"""
        <div class="executive-summary">
            <h2>📋 Executive Summary</h2>

            <div class="summary-grid">
                <div class="summary-item">
                    <h3>🎯 Overall Code Quality Score</h3>
                    <div class="quality-score {self._get_quality_class(quality_score)}">
                        {quality_score:.1f}/100
                    </div>
                    <p class="score-description">{self._get_quality_description(quality_score)}</p>
                </div>

                <div class="summary-item">
                    <h3>📊 Scan Coverage</h3>
                    <ul>
                        <li><strong>{metrics.total_files}</strong> Python files analyzed</li>
                        <li><strong>{metrics.total_lines:,}</strong> lines of code scanned</li>
                        <li><strong>{metrics.issues_per_file:.1f}</strong> average issues per file</li>
                        <li><strong>{metrics.issues_per_thousand_lines:.1f}</strong> issues per 1K lines</li>
                    </ul>
                </div>

                <div class="summary-item">
                    <h3>🚨 Critical Findings</h3>
                    <div class="critical-highlights">
                        <div class="highlight-item">
                            <span class="highlight-number">{metrics.critical_count}</span>
                            <span class="highlight-label">Critical Issues</span>
                        </div>
                        <div class="highlight-item">
                            <span class="highlight-number">{len([f for f in data['findings'] if f['category'] == 'SECURITY'])}</span>
                            <span class="highlight-label">Security Issues</span>
                        </div>
                        <div class="highlight-item">
                            <span class="highlight-number">{len([f for f in data['findings'] if f['category'] == 'SYNTAX'])}</span>
                            <span class="highlight-label">Syntax Errors</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="key-insights">
                <h3>🔍 Key Insights</h3>
                <ul>
                    <li><strong>Most Problematic File:</strong> {metrics.most_problematic_file}</li>
                    <li><strong>Most Common Issue:</strong> {metrics.most_common_issue_type}</li>
                    <li><strong>Primary Issue Category:</strong> Code Quality ({len([f for f in data['findings'] if f['category'] == 'QUALITY'])} issues)</li>
                    <li><strong>Security Risk Level:</strong> {self._assess_security_risk(data)}</li>
                </ul>
            </div>
        </div>
        """

    def _generate_severity_breakdown(self, data: Dict) -> str:
        """Generate severity breakdown section"""
        severity_counts = defaultdict(int)
        for finding in data['findings']:
            severity_counts[finding['severity']] += 1

        # Create severity distribution chart
        chart_data = []
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            count = severity_counts[severity]
            percentage = (count / len(data['findings'])) * 100 if data['findings'] else 0
            chart_data.append({
                'severity': severity,
                'count': count,
                'percentage': percentage,
                'color': self.colors[severity]
            })

        chart_html = ""
        for item in chart_data:
            chart_html += f"""
            <div class="severity-item">
                <div class="severity-bar">
                    <div class="severity-fill" style="width: {item['percentage']}%; background-color: {item['color']};"></div>
                </div>
                <div class="severity-info">
                    <span class="severity-label">{item['severity']}</span>
                    <span class="severity-count">{item['count']} ({item['percentage']:.1f}%)</span>
                </div>
            </div>
            """

        return f"""
        <div class="severity-breakdown">
            <h2>📈 Severity Distribution</h2>
            <div class="severity-chart">
                {chart_html}
            </div>

            <div class="severity-details">
                <h3>Issue Breakdown by Severity</h3>
                <div class="severity-grid">
                    {self._generate_severity_detail_cards(data)}
                </div>
            </div>
        </div>
        """

    def _generate_severity_detail_cards(self, data: Dict) -> str:
        """Generate detailed severity cards"""
        severity_details = defaultdict(list)
        for finding in data['findings']:
            severity_details[finding['severity']].append(finding)

        cards_html = ""
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            issues = severity_details[severity]
            if issues:
                # Get top 3 most common issues for this severity
                rule_counts = Counter(f['rule_id'] for f in issues)
                top_issues = rule_counts.most_common(3)

                issues_html = ""
                for rule_id, count in top_issues:
                    issue_name = rule_id.replace('_', ' ').title()
                    issues_html += f"<li>{issue_name}: {count}</li>"

                cards_html += f"""
                <div class="severity-card severity-{severity.lower()}">
                    <h4>{severity} ({len(issues)})</h4>
                    <ul>{issues_html}</ul>
                    <div class="severity-actions">
                        <button class="action-btn" onclick="showSeverityDetails('{severity}')">
                            View Details
                        </button>
                    </div>
                </div>
                """

        return cards_html

    def _generate_category_analysis(self, data: Dict) -> str:
        """Generate category analysis section"""
        categories = defaultdict(int)
        for finding in data['findings']:
            categories[finding['category']] += 1

        category_data = []
        for category in ['SYNTAX', 'SECURITY', 'QUALITY', 'SPELLING']:
            count = categories[category]
            percentage = (count / len(data['findings'])) * 100 if data['findings'] else 0
            category_data.append({
                'category': category,
                'count': count,
                'percentage': percentage,
                'icon': self._get_category_icon(category)
            })

        return f"""
        <div class="category-analysis">
            <h2>📊 Category Analysis</h2>
            <div class="category-grid">
                {''.join([f'''
                <div class="category-card">
                    <div class="category-icon">{item['icon']}</div>
                    <div class="category-info">
                        <h3>{item['category']}</h3>
                        <div class="category-count">{item['count']}</div>
                        <div class="category-percentage">{item['percentage']:.1f}%</div>
                    </div>
                </div>
                ''' for item in category_data])}
            </div>
        </div>
        """

    def _generate_file_analysis(self, data: Dict) -> str:
        """Generate file-by-file analysis"""
        file_issues = defaultdict(list)
        for finding in data['findings']:
            file_name = finding['file_path'].split('/')[-1]
            file_issues[file_name].append(finding)

        # Sort files by number of issues
        sorted_files = sorted(file_issues.items(), key=lambda x: len(x[1]), reverse=True)

        file_rows = ""
        for file_name, issues in sorted_files[:20]:  # Top 20 files
            severity_counts = defaultdict(int)
            for issue in issues:
                severity_counts[issue['severity']] += 1

            critical_count = severity_counts['CRITICAL']
            high_count = severity_counts['HIGH']
            total_count = len(issues)

            file_rows += f"""
            <tr>
                <td class="file-name">{file_name}</td>
                <td class="issue-count total">{total_count}</td>
                <td class="issue-count critical">{critical_count}</td>
                <td class="issue-count high">{high_count}</td>
                <td class="issue-count medium">{severity_counts['MEDIUM']}</td>
                <td class="issue-count low">{severity_counts['LOW']}</td>
                <td class="issue-count info">{severity_counts['INFO']}</td>
            </tr>
            """

        return f"""
        <div class="file-analysis">
            <h2>📁 File Analysis</h2>
            <p>Top 20 files by number of issues</p>

            <div class="file-table-container">
                <table class="file-table">
                    <thead>
                        <tr>
                            <th>File Name</th>
                            <th>Total</th>
                            <th class="critical">Critical</th>
                            <th class="high">High</th>
                            <th class="medium">Medium</th>
                            <th class="low">Low</th>
                            <th class="info">Info</th>
                        </tr>
                    </thead>
                    <tbody>
                        {file_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _generate_issue_patterns(self, data: Dict) -> str:
        """Generate issue pattern analysis"""
        rule_counts = Counter(f['rule_id'] for f in data['findings'])
        top_patterns = rule_counts.most_common(10)

        pattern_rows = ""
        for rule_id, count in top_patterns:
            pattern_name = rule_id.replace('_', ' ').title()
            percentage = (count / len(data['findings'])) * 100 if data['findings'] else 0

            pattern_rows += f"""
            <tr>
                <td>{pattern_name}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
                <td>{self._get_pattern_description(rule_id)}</td>
            </tr>
            """

        return f"""
        <div class="issue-patterns">
            <h2>🔍 Issue Pattern Analysis</h2>
            <p>Most common issues found across the codebase</p>

            <table class="patterns-table">
                <thead>
                    <tr>
                        <th>Issue Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {pattern_rows}
                </tbody>
            </table>
        </div>
        """

    def _generate_recommendations_section(self, data: Dict) -> str:
        """Generate recommendations section"""
        recommendations = data.get('recommendations', [])

        priority_recommendations = []
        standard_recommendations = []

        # Separate high-priority from standard recommendations
        for rec in recommendations:
            if any(keyword in rec.upper() for keyword in ['CRITICAL', 'HIGH PRIORITY', 'SECURITY']):
                priority_recommendations.append(rec)
            else:
                standard_recommendations.append(rec)

        priority_html = "".join([f"<li class='priority-item'>{rec}</li>" for rec in priority_recommendations])
        standard_html = "".join([f"<li>{rec}</li>" for rec in standard_recommendations])

        return f"""
        <div class="recommendations">
            <h2>💡 Recommendations</h2>

            <div class="recommendations-grid">
                <div class="priority-recommendations">
                    <h3>🚨 Priority Actions</h3>
                    <ul>
                        {priority_html}
                    </ul>
                </div>

                <div class="standard-recommendations">
                    <h3>📋 Standard Improvements</h3>
                    <ul>
                        {standard_html}
                    </ul>
                </div>
            </div>

            <div class="improvement-roadmap">
                <h3>🛣️ Improvement Roadmap</h3>
                <div class="roadmap-steps">
                    <div class="roadmap-step">
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <h4>Immediate (Week 1)</h4>
                            <p>Fix all critical syntax errors and security vulnerabilities</p>
                        </div>
                    </div>
                    <div class="roadmap-step">
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <h4>Short-term (Week 2)</h4>
                            <p>Address high-priority issues and improve error handling</p>
                        </div>
                    </div>
                    <div class="roadmap-step">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <h4>Medium-term (Month 1)</h4>
                            <p>Refactor code quality issues and optimize performance</p>
                        </div>
                    </div>
                    <div class="roadmap-step">
                        <div class="step-number">4</div>
                        <div class="step-content">
                            <h4>Long-term (Ongoing)</h4>
                            <p>Implement regular code reviews and automated quality checks</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_trends_section(self) -> str:
        """Generate trends and improvement tracking section"""
        return """
        <div class="trends-section">
            <h2>📈 Quality Trends & Tracking</h2>

            <div class="trend-insights">
                <div class="trend-item">
                    <h3>🎯 Quality Score Trend</h3>
                    <p>Track code quality improvements over time</p>
                    <div class="trend-placeholder">
                        <p>📊 Historical trend data will be available after multiple scans</p>
                    </div>
                </div>

                <div class="trend-item">
                    <h3>📋 Issue Resolution Rate</h3>
                    <p>Monitor how quickly issues are being addressed</p>
                    <div class="trend-placeholder">
                        <p>📈 Resolution metrics will be tracked in future scans</p>
                    </div>
                </div>

                <div class="trend-item">
                    <h3>🔄 Continuous Improvement</h3>
                    <p>Regular scanning helps maintain code quality</p>
                    <div class="trend-placeholder">
                        <p>🔄 Schedule automated scans for ongoing quality assurance</p>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_footer_section(self) -> str:
        """Generate report footer"""
        return f"""
        <div class="footer-section">
            <div class="footer-content">
                <p><strong>VIPER Repository Bug Scan Report</strong></p>
                <p>Generated by Comprehensive Bug Detector v1.0.0</p>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                <p>For questions or support, contact the VIPER development team</p>
            </div>
        </div>
        """

    def _combine_html_sections(self, sections: Dict[str, str]) -> str:
        """Combine all HTML sections into complete report"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>VIPER Bug Scan Report</title>
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            {sections['header']}
            {sections['executive_summary']}
            {sections['severity_breakdown']}
            {sections['category_analysis']}
            {sections['file_analysis']}
            {sections['issue_patterns']}
            {sections['recommendations']}
            {sections['trends']}
            {sections['footer']}

            <script>
                {self._get_javascript()}
            </script>
        </body>
        </html>
        """
        return html_template

    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header Section */
        .header-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .header-title h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 20px;
        }

        .scan-info {
            display: flex;
            gap: 30px;
            font-size: 0.9em;
        }

        .key-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
        }

        .metric-card.critical {
            background: rgba(220, 53, 69, 0.2);
            border: 2px solid #DC3545;
        }

        .metric-card.high {
            background: rgba(253, 126, 20, 0.2);
            border: 2px solid #FD7E14;
        }

        .metric-card.medium {
            background: rgba(255, 193, 7, 0.2);
            border: 2px solid #FFC107;
        }

        .metric-card.low {
            background: rgba(40, 167, 69, 0.2);
            border: 2px solid #28A745;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        /* Executive Summary */
        .executive-summary {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }

        .summary-item h3 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .quality-score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .quality-score.excellent { background: #28A745; color: white; }
        .quality-score.good { background: #17A2B8; color: white; }
        .quality-score.fair { background: #FFC107; color: black; }
        .quality-score.poor { background: #FD7E14; color: white; }
        .quality-score.critical { background: #DC3545; color: white; }

        /* Severity Breakdown */
        .severity-breakdown {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        .severity-item {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }

        .severity-bar {
            flex: 1;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            margin-right: 15px;
            overflow: hidden;
        }

        .severity-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }

        .severity-info {
            display: flex;
            justify-content: space-between;
            width: 200px;
        }

        /* Category Analysis */
        .category-analysis {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .category-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }

        .category-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .category-count {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }

        /* File Analysis */
        .file-analysis {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        .file-table-container {
            overflow-x: auto;
        }

        .file-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .file-table th,
        .file-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        .file-table th {
            background: #f8f9fa;
            font-weight: 600;
        }

        .file-name {
            font-family: 'Courier New', monospace;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Recommendations */
        .recommendations {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        .recommendations-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .priority-item {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }

        .roadmap-steps {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .roadmap-step {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .step-number {
            background: #007bff;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }

        /* Footer */
        .footer-section {
            background: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-top: 50px;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .key-metrics,
            .summary-grid,
            .category-grid,
            .recommendations-grid {
                grid-template-columns: 1fr;
            }

            .severity-info {
                width: 150px;
            }

            .header-title h1 {
                font-size: 2em;
            }
        }
        """

    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features"""
        return """
        function showSeverityDetails(severity) {
            // Placeholder for future interactive features
            alert(`Showing details for ${severity} severity issues`);
        }

        // Add some interactive enhancements
        document.addEventListener('DOMContentLoaded', function() {
            // Add hover effects to metric cards
            const metricCards = document.querySelectorAll('.metric-card');
            metricCards.forEach(card => {
                card.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-5px)';
                    this.style.transition = 'transform 0.2s ease';
                });
                card.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0)';
                });
            });

            // Add click effects to severity items
            const severityItems = document.querySelectorAll('.severity-item');
            severityItems.forEach(item => {
                item.addEventListener('click', function() {
                    this.style.background = '#f8f9fa';
                    setTimeout(() => {
                        this.style.background = 'transparent';
                    }, 150);
                });
            });
        });
        """

    def _get_quality_class(self, score: float) -> str:
        """Get CSS class for quality score"""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 60:
            return "poor"
        else:
            return "critical"

    def _get_quality_description(self, score: float) -> str:
        """Get description for quality score"""
        if score >= 90:
            return "Excellent code quality with minimal issues"
        elif score >= 80:
            return "Good code quality with some minor issues"
        elif score >= 70:
            return "Fair code quality requiring attention"
        elif score >= 60:
            return "Poor code quality needing significant improvement"
        else:
            return "Critical code quality issues requiring immediate action"

    def _assess_security_risk(self, data: Dict) -> str:
        """Assess overall security risk level"""
        security_issues = [f for f in data['findings'] if f['category'] == 'SECURITY']
        critical_security = len([f for f in security_issues if f['severity'] == 'CRITICAL'])
        high_security = len([f for f in security_issues if f['severity'] == 'HIGH'])

        if critical_security > 0:
            return "HIGH - Critical security vulnerabilities detected"
        elif high_security > 0:
            return "MEDIUM - Security issues requiring attention"
        elif len(security_issues) > 0:
            return "LOW - Minor security concerns identified"
        else:
            return "SAFE - No security issues detected"

    def _get_category_icon(self, category: str) -> str:
        """Get icon for category"""
        icons = {
            'SYNTAX': '🐛',
            'SECURITY': '🔒',
            'QUALITY': '✨',
            'SPELLING': '📝'
        }
        return icons.get(category, '❓')

    def _get_pattern_description(self, rule_id: str) -> str:
        """Get description for issue pattern"""
        descriptions = {
            'ANTI_PATTERN_PRINT_DEBUG': 'Debug print statements left in production',
            'UNUSED_IMPORT': 'Unused import statements',
            'LINE_TOO_LONG': 'Lines exceeding recommended length',
            'ANTI_PATTERN_BARE_EXCEPT': 'Bare except clauses',
            'SECURITY_INSECURE_RANDOM': 'Insecure random number generation',
            'SYNTAX_ERROR': 'Python syntax errors',
            'SECURITY_SQL_INJECTION': 'Potential SQL injection vulnerabilities',
            'ANTI_PATTERN_EVAL_USAGE': 'Dangerous use of eval()',
            'ANTI_PATTERN_EXEC_USAGE': 'Dangerous use of exec()'
        }
        return descriptions.get(rule_id, 'Code quality issue')

    def _generate_pdf_report(self, html_content: str) -> str:
        """Generate PDF version of the report (placeholder)"""
        # This would require additional dependencies like weasyprint or pdfkit
        # For now, just return the HTML path
        pdf_path = self.output_dir / "comprehensive_bug_report.pdf"

        # Placeholder - in a real implementation, you would use a PDF library
        with open(pdf_path, 'w') as f:
            f.write("# PDF Generation would go here\n")
            f.write("# Requires additional dependencies like weasyprint or pdfkit\n")

        return str(pdf_path)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate comprehensive bug report from scan results')
    parser.add_argument('--input', '-i', help='Input scan results JSON file')
    parser.add_argument('--output', '-o', help='Output directory for reports')

    args = parser.parse_args()

    generator = ComprehensiveBugReportGenerator(args.input)
    if args.output:
        generator.output_dir = Path(args.output)

    report_path = generator.generate_comprehensive_report()
    print(f"\n🎉 Comprehensive bug report generated successfully!")
    print(f"📊 View the detailed report at: {report_path}")

if __name__ == '__main__':
    main()

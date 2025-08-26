#!/usr/bin/env python3
"""
🧪 VIPER Trading Bot - Comprehensive MCP Servers Test Suite
Test all MCP servers to ensure they are properly configured and operational
"""

import os
import sys
import asyncio
import time
import json
from typing import Dict, Any, List
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPTestSuite:
    """Comprehensive test suite for all MCP servers"""
    
    def __init__(self):
        self.test_results = {}
        self.mcp_servers = {
            "viper-trading-system": {
                "port": 8000,
                "description": "VIPER Trading System MCP Server",
                "endpoints": ["/", "/health", "/trading/status", "/microservices/status"]
            },
            "github-project-manager": {
                "port": 8001,
                "description": "GitHub Project Management MCP Server",
                "endpoints": ["/", "/health", "/github/tasks", "/github/projects"]
            },
            "trading-optimizer": {
                "port": 8002,
                "description": "Trading Strategy Optimizer MCP Server",
                "endpoints": ["/", "/health", "/metrics/performance", "/optimization/history"]
            }
        }
    
    async def test_server_connectivity(self, server_name: str, port: int) -> Dict[str, Any]:
        """Test basic server connectivity"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/", timeout=5)
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "connectivity": "reachable",
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "response_data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200]
                    }
                else:
                    return {
                        "status": "error",
                        "connectivity": "reachable",
                        "error": f"HTTP {response.status_code}",
                        "response_text": response.text[:200]
                    }
                    
        except httpx.ConnectError:
            return {
                "status": "error",
                "connectivity": "unreachable",
                "error": "Connection refused - server may not be running"
            }
        except httpx.TimeoutException:
            return {
                "status": "error",
                "connectivity": "timeout",
                "error": "Request timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "connectivity": "error",
                "error": str(e)
            }
    
    async def test_server_health(self, server_name: str, port: int) -> Dict[str, Any]:
        """Test server health endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/health", timeout=5)
                
                if response.status_code == 200:
                    health_data = response.json()
                    return {
                        "status": "success",
                        "health_check": "passed",
                        "health_data": health_data,
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                else:
                    return {
                        "status": "error",
                        "health_check": "failed",
                        "error": f"HTTP {response.status_code}",
                        "response_text": response.text[:200]
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "health_check": "failed",
                "error": str(e)
            }
    
    async def test_server_endpoints(self, server_name: str, port: int, endpoints: List[str]) -> Dict[str, Any]:
        """Test all server endpoints"""
        endpoint_results = {}
        
        for endpoint in endpoints:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://localhost:{port}{endpoint}", timeout=5)
                    
                    if response.status_code == 200:
                        endpoint_results[endpoint] = {
                            "status": "success",
                            "response_code": response.status_code,
                            "response_time_ms": response.elapsed.total_seconds() * 1000
                        }
                    else:
                        endpoint_results[endpoint] = {
                            "status": "error",
                            "response_code": response.status_code,
                            "error": f"HTTP {response.status_code}"
                        }
                        
            except Exception as e:
                endpoint_results[endpoint] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "status": "completed",
            "endpoints_tested": len(endpoints),
            "endpoint_results": endpoint_results
        }
    
    async def test_github_integration(self) -> Dict[str, Any]:
        """Test GitHub integration functionality"""
        try:
            # Check if GitHub environment variables are configured
            github_pat = os.getenv('GITHUB_PAT')
            github_owner = os.getenv('GITHUB_OWNER')
            github_repo = os.getenv('GITHUB_REPO')
            
            if not all([github_pat, github_owner, github_repo]):
                return {
                    "status": "warning",
                    "github_integration": "not_configured",
                    "missing_vars": [var for var, val in [('GITHUB_PAT', github_pat), ('GITHUB_OWNER', github_owner), ('GITHUB_REPO', github_repo)] if not val]
                }
            
            # Test GitHub API connectivity
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"token {github_pat}",
                    "Accept": "application/vnd.github.v3+json"
                }
                
                # Test repository access
                repo_url = f"https://api.github.com/repos/{github_owner}/{github_repo}"
                response = await client.get(repo_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    repo_data = response.json()
                    return {
                        "status": "success",
                        "github_integration": "working",
                        "repository": f"{github_owner}/{github_repo}",
                        "repository_name": repo_data.get("name"),
                        "description": repo_data.get("description"),
                        "stars": repo_data.get("stargazers_count"),
                        "forks": repo_data.get("forks_count")
                    }
                else:
                    return {
                        "status": "error",
                        "github_integration": "api_error",
                        "error": f"GitHub API returned {response.status_code}",
                        "response_text": response.text[:200]
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "github_integration": "connection_error",
                "error": str(e)
            }
    
    async def test_trading_optimization(self) -> Dict[str, Any]:
        """Test trading optimization functionality"""
        try:
            # Test optimization endpoints
            async with httpx.AsyncClient() as client:
                # Test performance optimization
                perf_response = await client.post(
                    "http://localhost:8002/optimize/performance",
                    json={"service": "api-server", "time_range": "1h"},
                    timeout=10
                )
                
                if perf_response.status_code == 200:
                    return {
                        "status": "success",
                        "trading_optimization": "working",
                        "performance_optimization": "successful",
                        "response_data": perf_response.json()
                    }
                else:
                    return {
                        "status": "error",
                        "trading_optimization": "failed",
                        "error": f"Performance optimization returned {perf_response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "trading_optimization": "connection_error",
                "error": str(e)
            }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite for all MCP servers"""
        print("🧪 Starting comprehensive MCP servers test suite...")
        print("="*80)
        
        start_time = time.time()
        
        # Test each server
        for server_name, config in self.mcp_servers.items():
            print(f"\n🔍 Testing {server_name}...")
            print(f"   📝 {config['description']}")
            print(f"   🌐 Port: {config['port']}")
            
            server_results = {}
            
            # Test connectivity
            print("   🔌 Testing connectivity...")
            connectivity_result = await self.test_server_connectivity(server_name, config['port'])
            server_results['connectivity'] = connectivity_result
            
            if connectivity_result['status'] == 'success':
                print("      ✅ Server is reachable")
                
                # Test health endpoint
                print("   💚 Testing health endpoint...")
                health_result = await self.test_server_health(server_name, config['port'])
                server_results['health'] = health_result
                
                if health_result['status'] == 'success':
                    print("      ✅ Health check passed")
                else:
                    print(f"      ❌ Health check failed: {health_result.get('error', 'Unknown error')}")
                
                # Test all endpoints
                print("   🌐 Testing endpoints...")
                endpoints_result = await self.test_server_endpoints(server_name, config['port'], config['endpoints'])
                server_results['endpoints'] = endpoints_result
                
                successful_endpoints = sum(1 for ep_result in endpoints_result['endpoint_results'].values() if ep_result['status'] == 'success')
                print(f"      📊 {successful_endpoints}/{len(config['endpoints'])} endpoints working")
                
            else:
                print(f"      ❌ Server is not reachable: {connectivity_result.get('error', 'Unknown error')}")
            
            self.test_results[server_name] = server_results
        
        # Test GitHub integration
        print("\n🔍 Testing GitHub integration...")
        github_result = await self.test_github_integration()
        self.test_results['github_integration'] = github_result
        
        if github_result['status'] == 'success':
            print("   ✅ GitHub integration working")
        elif github_result['status'] == 'warning':
            print(f"   ⚠️  GitHub integration not fully configured: {', '.join(github_result.get('missing_vars', []))}")
        else:
            print(f"   ❌ GitHub integration failed: {github_result.get('error', 'Unknown error')}")
        
        # Test trading optimization
        print("\n🔍 Testing trading optimization...")
        optimization_result = await self.test_trading_optimization()
        self.test_results['trading_optimization'] = optimization_result
        
        if optimization_result['status'] == 'success':
            print("   ✅ Trading optimization working")
        else:
            print(f"   ❌ Trading optimization failed: {optimization_result.get('error', 'Unknown error')}")
        
        # Calculate test summary
        total_tests = len(self.mcp_servers) + 2  # +2 for GitHub and optimization tests
        successful_tests = 0
        
        for server_name, results in self.test_results.items():
            if server_name in self.mcp_servers:
                # Server test
                if results.get('connectivity', {}).get('status') == 'success':
                    successful_tests += 1
            else:
                # Integration test
                if results.get('status') == 'success':
                    successful_tests += 1
        
        test_duration = time.time() - start_time
        
        # Generate comprehensive report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate_percent": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                "test_duration_seconds": test_duration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check server connectivity
        for server_name, results in self.test_results.items():
            if server_name in self.mcp_servers:
                if results.get('connectivity', {}).get('status') != 'success':
                    recommendations.append(f"Start {server_name} server - it's not reachable")
                elif results.get('health', {}).get('status') != 'success':
                    recommendations.append(f"Check {server_name} server health - health check failed")
        
        # Check GitHub integration
        github_result = self.test_results.get('github_integration', {})
        if github_result.get('status') == 'warning':
            missing_vars = github_result.get('missing_vars', [])
            recommendations.append(f"Configure missing GitHub environment variables: {', '.join(missing_vars)}")
        elif github_result.get('status') == 'error':
            recommendations.append("Fix GitHub integration issues - check API access and permissions")
        
        # Check trading optimization
        optimization_result = self.test_results.get('trading_optimization', {})
        if optimization_result.get('status') != 'success':
            recommendations.append("Ensure trading optimizer server is running and accessible")
        
        if not recommendations:
            recommendations.append("All systems are operational - no immediate action required")
        
        return recommendations
    
    def print_test_report(self, report: Dict[str, Any]):
        """Print comprehensive test report"""
        print("\n" + "="*80)
        print("📊 MCP SERVERS COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        summary = report['test_summary']
        print(f"🧪 Test Summary:")
        print(f"   📊 Total Tests: {summary['total_tests']}")
        print(f"   ✅ Successful: {summary['successful_tests']}")
        print(f"   📈 Success Rate: {summary['success_rate_percent']:.1f}%")
        print(f"   ⏱️  Duration: {summary['test_duration_seconds']:.2f} seconds")
        print(f"   🕐 Timestamp: {summary['timestamp']}")
        
        print(f"\n🔍 Detailed Results:")
        for server_name, results in report['detailed_results'].items():
            if server_name in self.mcp_servers:
                print(f"\n   🖥️  {server_name}:")
                connectivity = results.get('connectivity', {})
                if connectivity.get('status') == 'success':
                    print(f"      ✅ Connectivity: {connectivity.get('response_time_ms', 0):.1f}ms")
                    
                    health = results.get('health', {})
                    if health.get('status') == 'success':
                        print(f"      ✅ Health: {health.get('health_data', {}).get('status', 'unknown')}")
                    else:
                        print(f"      ❌ Health: {health.get('error', 'Unknown error')}")
                    
                    endpoints = results.get('endpoints', {})
                    successful_eps = sum(1 for ep_result in endpoints.get('endpoint_results', {}).values() if ep_result.get('status') == 'success')
                    total_eps = len(endpoints.get('endpoint_results', {}))
                    print(f"      🌐 Endpoints: {successful_eps}/{total_eps} working")
                else:
                    print(f"      ❌ Connectivity: {connectivity.get('error', 'Unknown error')}")
            else:
                print(f"\n   🔗 {server_name}:")
                if results.get('status') == 'success':
                    print(f"      ✅ Working correctly")
                elif results.get('status') == 'warning':
                    print(f"      ⚠️  {results.get('error', 'Warning')}")
                else:
                    print(f"      ❌ {results.get('error', 'Unknown error')}")
        
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"   {i}. {recommendation}")
        
        print("\n" + "="*80)
        
        # Save report to file
        report_file = f"mcp_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Detailed report saved to: {report_file}")

async def main():
    """Main test function"""
    print("🚀 VIPER Trading Bot - MCP Servers Test Suite")
    print("="*60)
    
    test_suite = MCPTestSuite()
    
    try:
        report = await test_suite.run_comprehensive_tests()
        test_suite.print_test_report(report)
        
        # Exit with appropriate code
        success_rate = report['test_summary']['success_rate_percent']
        if success_rate >= 80:
            print("\n🎉 Test suite completed successfully!")
            sys.exit(0)
        elif success_rate >= 50:
            print("\n⚠️  Test suite completed with warnings - some issues detected")
            sys.exit(1)
        else:
            print("\n❌ Test suite failed - critical issues detected")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

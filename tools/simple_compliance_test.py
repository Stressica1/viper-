#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Simple Compliance Test
Tests critical microservices architecture rules
"""

import os
from pathlib import Path

def test_critical_rules():
    """Test the most critical microservices rules"""
    print("🚀 VIPER Critical Rules Compliance Test")
    print("=" * 40)

    project_root = Path(".")
    services_dir = project_root / "services"
    tests_passed = 0
    total_tests = 0

    # Test 1: Service structure
    total_tests += 1
    required_services = ['api-server', 'ultra-backtester', 'strategy-optimizer',
                        'live-trading-engine', 'data-manager', 'exchange-connector',
                        'risk-manager', 'monitoring-service']

    missing_services = []
    for service in required_services:
        if not (services_dir / service).exists():
            missing_services.append(service)

    if not missing_services:
        print("✅ Service Structure: All 8 services present")
        tests_passed += 1
    else:
        print(f"❌ Service Structure: Missing {missing_services}")

    # Test 2: Docker configuration
    total_tests += 1
    docker_compose = project_root / "docker-compose.yml"
    if docker_compose.exists():
        with open(docker_compose, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if 'version:' not in content:
            print("✅ Docker Config: No deprecated version field")
            tests_passed += 1
        else:
            print("❌ Docker Config: Deprecated version field found")
    else:
        print("❌ Docker Config: docker-compose.yml not found")

    # Test 3: Health checks
    total_tests += 1
    health_count = 0
    for service in required_services:
        main_py = services_dir / service / "main.py"
        if main_py.exists():
            try:
                with open(main_py, 'r', encoding='utf-8', errors='ignore') as f:
                    if 'def health_check' in f.read():
                        health_count += 1
            except:
                pass

    if health_count == len(required_services):
        print(f"✅ Health Checks: All {health_count} services have health endpoints")
        tests_passed += 1
    else:
        print(f"❌ Health Checks: Only {health_count}/{len(required_services)} services have health endpoints")

    # Test 4: Environment variables
    total_tests += 1
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        required_vars = ['BITGET_API_KEY', 'REDIS_URL', 'LOG_LEVEL']
        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)

        if not missing_vars:
            print("✅ Environment: All required variables configured")
            tests_passed += 1
        else:
            print(f"❌ Environment: Missing variables: {missing_vars}")

    # Test 5: Circuit breaker
    total_tests += 1
    circuit_breaker = services_dir / "shared" / "circuit_breaker.py"
    if circuit_breaker.exists():
        print("✅ Circuit Breaker: Implementation exists")
        tests_passed += 1
    else:
        print("❌ Circuit Breaker: Implementation missing")

    # Test 6: Service discovery
    total_tests += 1
    services_using_dns = 0
    for service in required_services:
        main_py = services_dir / service / "main.py"
        if main_py.exists():
            try:
                with open(main_py, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'http://' in content and 'localhost' not in content:
                        services_using_dns += 1
            except:
                pass

    if services_using_dns >= len(required_services) // 2:  # At least half
        print(f"✅ Service Discovery: {services_using_dns} services using DNS names")
        tests_passed += 1
    else:
        print(f"❌ Service Discovery: Only {services_using_dns} services using DNS names")

    print("\n" + "=" * 40)
    print(f"📊 COMPLIANCE SCORE: {tests_passed}/{total_tests} ({tests_passed/total_tests*100:.1f}%)")

    if tests_passed == total_tests:
        print("🎉 ALL CRITICAL RULES PASSED!")
        print("✅ System is microservices-compliant and ready for deployment")
        return True
    else:
        print("⚠️  SOME RULES FAILED!")
        print("❌ System needs fixes before production deployment")
        return False

if __name__ == "__main__":
    success = test_critical_rules()
    exit(0 if success else 1)

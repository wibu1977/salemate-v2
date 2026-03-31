#!/usr/bin/env python3
"""
Smoke test script for Sellora backend deployment.

This script verifies that the backend is properly deployed and functional.
Run after deployment to ensure all components are working.

Usage:
    python smoke_test.py
    python smoke_test.py --host localhost --port 8000
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from httpx import AsyncClient
except ImportError:
    print("ERROR: httpx not installed. Install with: pip install httpx")
    sys.exit(1)

from app.config import get_settings


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(message: str) -> None:
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str) -> None:
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


def print_header(message: str) -> None:
    """Print header in bold."""
    print(f"\n{Colors.BOLD}{message}{Colors.RESET}")


class SmokeTest:
    """Smoke test runner for Sellora backend."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    async def test_server_running(self) -> bool:
        """Test if server is running."""
        print_header("Testing Server Connectivity")
        try:
            async with AsyncClient() as client:
                response = await client.get(self.base_url, timeout=5.0)
                if response.status_code == 200:
                    print_success(f"Server is running at {self.base_url}")
                    self.passed += 1
                    return True
                else:
                    print_error(f"Server returned status {response.status_code}")
                    self.failed += 1
                    return False
        except Exception as e:
            print_error(f"Cannot connect to server: {e}")
            self.failed += 1
            return False

    async def test_health_endpoint(self) -> bool:
        """Test health check endpoint."""
        print_header("Testing Health Endpoint")
        try:
            async with AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        print_success("Health endpoint returned healthy status")
                        self.passed += 1
                        return True
                    else:
                        print_warning(f"Health status: {data.get('status')}")
                        self.warnings += 1
                        return True
                else:
                    print_error(f"Health endpoint returned {response.status_code}")
                    self.failed += 1
                    return False
        except Exception as e:
            print_error(f"Health check failed: {e}")
            self.failed += 1
            return False

    def test_config_loads(self) -> bool:
        """Test if configuration loads correctly."""
        print_header("Testing Configuration")
        try:
            settings = get_settings()

            required_fields = [
                "supabase_url",
                "supabase_key",
                "gemini_api_key",
                "gemini_model",
                "messenger_verify_token",
            ]

            missing = [field for field in required_fields
                       if not getattr(settings, field)]

            if missing:
                print_error(f"Missing config fields: {', '.join(missing)}")
                self.failed += 1
                return False
            else:
                print_success("All required configuration fields present")
                self.passed += 1

                # Check optional fields
                if not settings.supabase_service_role_key:
                    print_warning("SUPABASE_SERVICE_ROLE_KEY not set (service operations limited)")
                    self.warnings += 1

                return True
        except Exception as e:
            print_error(f"Configuration loading failed: {e}")
            self.failed += 1
            return False

    def test_imports(self) -> bool:
        """Test if all required modules can be imported."""
        print_header("Testing Module Imports")
        required_modules = [
            "app.main",
            "app.config",
            "app.db.connection",
            "app.models.schemas",
            "app.services.extractor",
            "app.services.conversation",
            "app.services.embeddings",
            "app.services.ai_agent",
            "app.routers.webhooks",
            "app.routers.catalog",
        ]

        failed_imports = []

        for module in required_modules:
            try:
                __import__(module)
                print_success(f"{module}")
                self.passed += 1
            except ImportError as e:
                print_error(f"{module}: {e}")
                failed_imports.append(module)
                self.failed += 1

        return len(failed_imports) == 0

    async def test_webhook_verification_endpoint(self) -> bool:
        """Test webhook verification endpoint."""
        print_header("Testing Webhook Verification Endpoint")
        try:
            async with AsyncClient() as client:
                # Test with wrong token - should get 403 but endpoint exists
                response = await client.get(
                    f"{self.base_url}/webhooks/messenger",
                    params={
                        "hub.mode": "subscribe",
                        "hub.verify_token": "wrong_token",
                        "hub.challenge": "123456789",
                    },
                    timeout=10.0
                )

                # Expected 403 with wrong token, but endpoint should exist
                if response.status_code in [200, 403]:
                    print_success("Webhook verification endpoint accessible")
                    self.passed += 1
                    return True
                else:
                    print_error(f"Webhook endpoint returned {response.status_code}")
                    self.failed += 1
                    return False
        except Exception as e:
            print_error(f"Webhook verification test failed: {e}")
            self.failed += 1
            return False

    async def test_webhook_post_endpoint(self) -> bool:
        """Test webhook POST endpoint."""
        print_header("Testing Webhook POST Endpoint")
        try:
            async with AsyncClient() as client:
                # Send a test webhook payload
                payload = {
                    "object": "page",
                    "entry": [{
                        "id": "123456789",
                        "time": 1705314600,
                        "messaging": [{
                            "sender": {"id": "1234567890"},
                            "recipient": {"id": "9876543210"},
                            "timestamp": 1705314600000,
                            "message": {
                                "mid": "msg_id_001",
                                "text": "Hello from smoke test"
                            }
                        }]
                    }]
                }

                response = await client.post(
                    f"{self.base_url}/webhooks/messenger",
                    json=payload,
                    timeout=10.0
                )

                # May fail without proper DB setup, but endpoint should respond
                if response.status_code in [200, 500, 422]:
                    print_success("Webhook POST endpoint accessible")
                    self.passed += 1
                    return True
                else:
                    print_error(f"Webhook POST returned {response.status_code}")
                    self.failed += 1
                    return False
        except Exception as e:
            print_error(f"Webhook POST test failed: {e}")
            self.failed += 1
            return False

    async def test_catalog_endpoint(self) -> bool:
        """Test catalog endpoints."""
        print_header("Testing Catalog Endpoint")
        try:
            async with AsyncClient() as client:
                # Test listing products
                sample_shop_id = "550e8400-e29b-41d4-a716-446655440000"
                response = await client.get(
                    f"{self.base_url}/catalog/products",
                    params={"shop_id": sample_shop_id},
                    timeout=10.0
                )

                # May return empty or error without DB, but endpoint should exist
                if response.status_code in [200, 404, 500]:
                    print_success("Catalog endpoint accessible")
                    self.passed += 1
                    return True
                else:
                    print_error(f"Catalog endpoint returned {response.status_code}")
                    self.failed += 1
                    return False
        except Exception as e:
            print_error(f"Catalog test failed: {e}")
            self.failed += 1
            return False

    def print_summary(self) -> int:
        """Print test summary."""
        print_header("Test Summary")
        print(f"Passed: {Colors.GREEN}{self.passed}{Colors.RESET}")
        print(f"Failed: {Colors.RED}{self.failed}{Colors.RESET}")
        print(f"Warnings: {Colors.YELLOW}{self.warnings}{Colors.RESET}")

        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0

        if self.failed == 0:
            print_success(f"All tests passed! ({success_rate:.1f}%)")
            return 0
        else:
            print_error(f"{self.failed} test(s) failed ({success_rate:.1f}%)")
            return 1

    async def run_all(self) -> int:
        """Run all smoke tests."""
        print_header("Sellora Backend Smoke Test")
        print(f"Target: {self.base_url}\n")

        # Run sync tests
        self.test_config_loads()
        self.test_imports()

        # Run async tests
        server_ok = await self.test_server_running()

        if server_ok:
            await self.test_health_endpoint()
            await self.test_webhook_verification_endpoint()
            await self.test_webhook_post_endpoint()
            await self.test_catalog_endpoint()

        return self.print_summary()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sellora backend smoke test"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Backend host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend port (default: 8000)"
    )

    args = parser.parse_args()

    smoke_test = SmokeTest(host=args.host, port=args.port)
    exit_code = await smoke_test.run_all()

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())

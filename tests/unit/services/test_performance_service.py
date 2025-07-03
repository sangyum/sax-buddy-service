import pytest
import os
from unittest.mock import AsyncMock, Mock
from src.repositories.performance_repository import PerformanceRepository
from src.services.performance_service import PerformanceService
from google.cloud.storage.bucket import Bucket

class TestPerformanceService:
    """Test Performance Service methods"""

    @pytest.fixture
    def mock_performance_repository(self):
        """Create mock PerformanceRepository"""
        repository = Mock(spec=PerformanceRepository)
        return repository
    
    @pytest.fixture
    def mock_storage_bucket(self):
        storage_bucket = Mock(spec=Bucket)
        storage_bucket.blob = Mock()
        return storage_bucket
    
    @pytest.fixture
    def performance_service(self, mock_performance_repository, mock_storage_bucket):
        """Create performance service with mock dependencies"""
        return PerformanceService(mock_performance_repository, mock_storage_bucket)
    
    @pytest.mark.asyncio
    async def test_upload_performance(self, performance_service: PerformanceService, mock_storage_bucket: Bucket):
        """Test performance upload to file store"""
        session_id = "12345"
        content_type = "audio/x-wav"
        contents = os.urandom(32)

        await performance_service.upload_performance(session_id=session_id, content_type=content_type, contents=contents)

        expected_blob_name = "12345.wav"
        mock_storage_bucket.blob.assert_called_once_with(blob_name=expected_blob_name)

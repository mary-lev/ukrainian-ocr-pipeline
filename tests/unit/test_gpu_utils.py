"""
Unit tests for GPU utilities
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.utils.gpu import (
    check_gpu_availability,
    optimize_for_device,
    get_optimal_batch_size,
    monitor_gpu_memory,
    setup_colab_gpu,
    is_colab_environment
)


class TestGPUUtilities(unittest.TestCase):
    """Test cases for GPU utility functions"""
    
    def test_check_gpu_availability(self):
        """Test GPU availability checking"""
        gpu_info = check_gpu_availability()
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(gpu_info, dict)
        
        required_keys = [
            'cuda_available', 
            'gpu_count', 
            'gpu_names', 
            'gpu_memory',
            'recommended_device'
        ]
        
        for key in required_keys:
            self.assertIn(key, gpu_info)
        
        # Check data types
        self.assertIsInstance(gpu_info['cuda_available'], bool)
        self.assertIsInstance(gpu_info['gpu_count'], int)
        self.assertIsInstance(gpu_info['gpu_names'], list)
        self.assertIsInstance(gpu_info['gpu_memory'], list)
        self.assertIsInstance(gpu_info['recommended_device'], str)
        
        # GPU count should match list lengths
        self.assertEqual(gpu_info['gpu_count'], len(gpu_info['gpu_names']))
        self.assertEqual(gpu_info['gpu_count'], len(gpu_info['gpu_memory']))
        
        # Recommended device should be valid
        self.assertIn(gpu_info['recommended_device'], ['cpu', 'cuda'])
    
    def test_optimize_for_device_cpu(self):
        """Test device optimization for CPU"""
        # Should not raise exception
        optimize_for_device('cpu')
    
    def test_optimize_for_device_cuda(self):
        """Test device optimization for CUDA"""
        # Should not raise exception even if CUDA not available
        optimize_for_device('cuda')
    
    def test_optimize_for_device_invalid(self):
        """Test device optimization with invalid device"""
        # Should handle gracefully
        optimize_for_device('invalid_device')
    
    def test_recommend_batch_size_cpu(self):
        """Test batch size recommendation for CPU"""
        batch_size = get_optimal_batch_size('cpu', model_type='trocr')
        
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 4)  # CPU batch sizes should be small
    
    def test_recommend_batch_size_cuda(self):
        """Test batch size recommendation for CUDA"""
        batch_size = get_optimal_batch_size('cuda', model_type='trocr')
        
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        
        # With 8GB GPU memory, should get decent batch size
        self.assertGreaterEqual(batch_size, 1)
    
    def test_recommend_batch_size_different_models(self):
        """Test batch size recommendation for different model types"""
        model_types = ['trocr', 'ner', 'segmentation', 'unknown']
        
        for model_type in model_types:
            cpu_batch = get_optimal_batch_size('cpu', model_type=model_type)
            cuda_batch = get_optimal_batch_size('cuda', model_type=model_type)
            
            self.assertIsInstance(cpu_batch, int)
            self.assertIsInstance(cuda_batch, int)
            self.assertGreater(cpu_batch, 0)
            self.assertGreater(cuda_batch, 0)
    
    def test_monitor_gpu_memory(self):
        """Test GPU memory monitoring"""
        memory_stats = monitor_gpu_memory()
        
        self.assertIsInstance(memory_stats, dict)
        
        if 'error' in memory_stats:
            # CUDA not available
            self.assertEqual(memory_stats['error'], 'CUDA not available')
        else:
            # CUDA available - check structure
            for gpu_id, stats in memory_stats.items():
                self.assertIsInstance(stats, dict)
                
                required_keys = ['allocated', 'reserved', 'total']
                for key in required_keys:
                    self.assertIn(key, stats)
                    self.assertIsInstance(stats[key], (int, float))
                    self.assertGreaterEqual(stats[key], 0)
    
    def test_is_colab_environment(self):
        """Test Colab environment detection"""
        is_colab = is_colab_environment()
        
        self.assertIsInstance(is_colab, bool)
        
        # In normal testing environment, should be False
        self.assertFalse(is_colab)
    
    def test_setup_colab_gpu(self):
        """Test Colab GPU setup"""
        gpu_info = setup_colab_gpu()
        
        # Should return valid GPU info regardless of environment
        self.assertIsInstance(gpu_info, dict)
        
        required_keys = [
            'cuda_available',
            'gpu_count', 
            'gpu_names',
            'gpu_memory',
            'recommended_device'
        ]
        
        for key in required_keys:
            self.assertIn(key, gpu_info)
    
    def test_batch_size_scaling_with_memory(self):
        """Test that batch size scales appropriately with GPU memory"""
        # Test different memory sizes
        memory_sizes = [2.0, 4.0, 8.0, 16.0, 24.0]
        batch_sizes = []
        
        for memory_gb in memory_sizes:
            batch_size = get_optimal_batch_size(
                'cuda', 
                model_type='trocr'
            )
            batch_sizes.append(batch_size)
        
        # Generally, batch size should increase or stay same with more memory
        for i in range(1, len(batch_sizes)):
            self.assertGreaterEqual(batch_sizes[i], batch_sizes[i-1])
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Very small GPU memory
        small_batch = get_optimal_batch_size('cuda')
        self.assertGreater(small_batch, 0)
        
        # Very large GPU memory
        large_batch = get_optimal_batch_size('cuda')
        self.assertGreater(large_batch, 0)
        
        # Zero memory (should fallback)
        zero_batch = get_optimal_batch_size('cuda')
        self.assertGreater(zero_batch, 0)
        
        # Negative memory (should handle gracefully)
        neg_batch = get_optimal_batch_size('cuda')
        self.assertGreater(neg_batch, 0)


class TestGPUUtilityIntegration(unittest.TestCase):
    """Integration tests for GPU utilities"""
    
    def test_gpu_workflow(self):
        """Test complete GPU detection and optimization workflow"""
        # 1. Check GPU availability
        gpu_info = check_gpu_availability()
        device = gpu_info['recommended_device']
        
        # 2. Optimize for detected device
        optimize_for_device(device)
        
        # 3. Get recommended batch size
        batch_size = get_optimal_batch_size(device, model_type='trocr')
        
        self.assertGreater(batch_size, 0)
        self.assertIn(device, ['cpu', 'cuda'])
    
    def test_colab_workflow(self):
        """Test Colab-specific workflow"""
        # Setup GPU for Colab
        gpu_info = setup_colab_gpu()
        
        # Should provide complete GPU information
        self.assertIsInstance(gpu_info, dict)
        self.assertIn('cuda_available', gpu_info)
        self.assertIn('recommended_device', gpu_info)
        
        # Get batch size for detected device
        device = gpu_info['recommended_device']
        batch_size = get_optimal_batch_size(device)
        
        self.assertGreater(batch_size, 0)
    
    def test_memory_monitoring_integration(self):
        """Test memory monitoring integration"""
        gpu_info = check_gpu_availability()
        
        if gpu_info['cuda_available']:
            # Monitor memory before and after optimization
            memory_before = monitor_gpu_memory()
            optimize_for_device('cuda')
            memory_after = monitor_gpu_memory()
            
            # Both should return valid results
            self.assertIsInstance(memory_before, dict)
            self.assertIsInstance(memory_after, dict)
        else:
            # Should handle gracefully when CUDA not available
            memory_stats = monitor_gpu_memory()
            self.assertIn('error', memory_stats)


if __name__ == '__main__':
    unittest.main()
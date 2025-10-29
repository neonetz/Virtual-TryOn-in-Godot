"""
Feature Extractor Benchmark Tool
=================================
Compares ORB, BRISK, and AKAZE feature extractors on the same pipeline.
Measures training time, inference speed, and classification accuracy.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.features.orb_extractor import ORBExtractor
from src.features.alternative_extractors import BRISKExtractor, AKAZEExtractor, SIFTExtractor
from src.features.bovw import BoVWEncoder
from src.detector import SVMClassifier
from src.training import Trainer


class FeatureBenchmark:
    """
    Benchmarks different feature extractors on the same dataset.
    
    Compares:
    - ORB (Oriented FAST and Rotated BRIEF)
    - BRISK (Binary Robust Invariant Scalable Keypoints)
    - AKAZE (Accelerated-KAZE)
    - SIFT (Scale-Invariant Feature Transform) [optional]
    """
    
    def __init__(
        self,
        pos_dir: str,
        neg_dir: str,
        output_dir: str = 'benchmark_results',
        k: int = 256,
        random_state: int = 42
    ):
        """
        Initialize benchmark.
        
        Args:
            pos_dir: Positive samples directory
            neg_dir: Negative samples directory
            output_dir: Where to save benchmark results
            k: Number of visual words for BoVW
            random_state: Random seed
        """
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.output_dir = output_dir
        self.k = k
        self.random_state = random_state
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
        
    def benchmark_extractor(
        self,
        extractor_name: str,
        extractor,
        include_sift: bool = False
    ) -> Dict[str, Any]:
        """
        Benchmark a single feature extractor.
        
        Args:
            extractor_name: Name (e.g., "ORB", "BRISK")
            extractor: Feature extractor instance
            include_sift: Whether to test SIFT (slow)
            
        Returns:
            Dictionary with benchmark results
        """
        print("\n" + "="*60)
        print(f"BENCHMARKING {extractor_name}")
        print("="*60)
        
        results = {
            'extractor': extractor_name,
            'k': self.k,
            'timings': {},
            'metrics': {}
        }
        
        # Create trainer with custom extractor
        trainer = Trainer(
            pos_dir=self.pos_dir,
            neg_dir=self.neg_dir,
            models_dir=os.path.join(self.output_dir, extractor_name.lower()),
            reports_dir=os.path.join(self.output_dir, extractor_name.lower(), 'reports'),
            random_state=self.random_state
        )
        
        # Replace ORB extractor with custom one
        trainer.orb_extractor = extractor
        
        # 1. Load dataset
        print("\n1. Loading dataset...")
        start = time.time()
        trainer.load_dataset()
        results['timings']['data_loading'] = time.time() - start
        
        # 2. Extract features
        print("\n2. Extracting features...")
        start = time.time()
        trainer.extract_features()
        results['timings']['feature_extraction'] = time.time() - start
        
        # Count descriptors
        total_desc = sum(
            len(d) if d is not None else 0 
            for d in trainer.train_descriptors
        )
        results['total_descriptors'] = total_desc
        print(f"   Total descriptors: {total_desc}")
        
        # 3. Build codebook
        print("\n3. Building codebook...")
        start = time.time()
        trainer.build_codebook(k=self.k, max_descriptors=200000)
        results['timings']['codebook_building'] = time.time() - start
        
        # 4. Encode features
        print("\n4. Encoding features...")
        start = time.time()
        trainer.encode_features()
        results['timings']['feature_encoding'] = time.time() - start
        
        # 5. Train SVM
        print("\n5. Training SVM...")
        start = time.time()
        train_metrics = trainer.train_svm(kernel='linear', C=1.0)
        results['timings']['svm_training'] = time.time() - start
        
        # 6. Evaluate
        print("\n6. Evaluating...")
        start = time.time()
        test_metrics = trainer.evaluate(save_plots=True)
        results['timings']['evaluation'] = time.time() - start
        
        # Store metrics
        results['metrics'] = test_metrics
        
        # Calculate total time
        results['timings']['total'] = sum(results['timings'].values())
        
        # Print summary
        print("\n" + "-"*60)
        print(f"{extractor_name} RESULTS")
        print("-"*60)
        print(f"Total time: {results['timings']['total']:.2f}s")
        print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
        print(f"Precision:  {test_metrics['precision']:.4f}")
        print(f"Recall:     {test_metrics['recall']:.4f}")
        print(f"F1 Score:   {test_metrics['f1']:.4f}")
        if 'roc_auc' in test_metrics and test_metrics['roc_auc']:
            print(f"ROC AUC:    {test_metrics['roc_auc']:.4f}")
        print("-"*60)
        
        return results
    
    def run_benchmark(self, include_sift: bool = False) -> Dict[str, Any]:
        """
        Run benchmark on all feature extractors.
        
        Args:
            include_sift: Include SIFT (patented, slower)
            
        Returns:
            Complete benchmark results
        """
        print("\n" + "="*60)
        print("FEATURE EXTRACTOR BENCHMARK")
        print("="*60)
        print(f"Dataset: {self.pos_dir}, {self.neg_dir}")
        print(f"Codebook size (k): {self.k}")
        print(f"Include SIFT: {include_sift}")
        print("="*60)
        
        # Benchmark ORB
        orb = ORBExtractor(n_features=500)
        self.results['ORB'] = self.benchmark_extractor('ORB', orb)
        
        # Benchmark BRISK
        brisk = BRISKExtractor(threshold=30)
        self.results['BRISK'] = self.benchmark_extractor('BRISK', brisk)
        
        # Benchmark AKAZE
        akaze = AKAZEExtractor(threshold=0.001)
        self.results['AKAZE'] = self.benchmark_extractor('AKAZE', akaze)
        
        # Benchmark SIFT (optional)
        if include_sift:
            try:
                sift = SIFTExtractor(n_features=500)
                self.results['SIFT'] = self.benchmark_extractor('SIFT', sift)
            except Exception as e:
                print(f"\nWarning: SIFT benchmark failed: {e}")
                print("SIFT may not be available or requires opencv-contrib-python")
        
        # Save results
        self.save_results()
        
        # Print comparison
        self.print_comparison()
        
        return self.results
    
    def save_results(self) -> None:
        """Save benchmark results to JSON."""
        output_path = os.path.join(self.output_dir, 'benchmark_results.json')
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\n✓ Results saved to {output_path}")
    
    def print_comparison(self) -> None:
        """Print comparison table."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Header
        print(f"{'Extractor':<12} {'Time(s)':<10} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'Descriptors':<12}")
        print("-"*80)
        
        # Data rows
        for name, result in self.results.items():
            time_total = result['timings']['total']
            accuracy = result['metrics']['accuracy']
            f1 = result['metrics']['f1']
            auc = result['metrics'].get('roc_auc', 0) or 0
            desc_count = result.get('total_descriptors', 0)
            
            print(f"{name:<12} {time_total:<10.2f} {accuracy:<10.4f} {f1:<10.4f} {auc:<10.4f} {desc_count:<12}")
        
        print("="*80)
        
        # Winner analysis
        best_accuracy = max(r['metrics']['accuracy'] for r in self.results.values())
        best_f1 = max(r['metrics']['f1'] for r in self.results.values())
        fastest = min(r['timings']['total'] for r in self.results.values())
        
        print("\nBest Performers:")
        for name, result in self.results.items():
            if result['metrics']['accuracy'] == best_accuracy:
                print(f"  🏆 Highest Accuracy: {name} ({best_accuracy:.4f})")
            if result['metrics']['f1'] == best_f1:
                print(f"  🏆 Highest F1 Score: {name} ({best_f1:.4f})")
            if result['timings']['total'] == fastest:
                print(f"  ⚡ Fastest Training: {name} ({fastest:.2f}s)")
        
        print("="*80)


def main():
    """Main benchmark script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark ORB vs BRISK vs AKAZE feature extractors'
    )
    parser.add_argument('--pos_dir', type=str, required=True,
                       help='Directory with positive samples')
    parser.add_argument('--neg_dir', type=str, required=True,
                       help='Directory with negative samples')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--k', type=int, default=256,
                       help='Number of visual words (codebook size)')
    parser.add_argument('--include_sift', action='store_true',
                       help='Include SIFT in benchmark (slower)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = FeatureBenchmark(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        output_dir=args.output_dir,
        k=args.k,
        random_state=args.seed
    )
    
    benchmark.run_benchmark(include_sift=args.include_sift)
    
    print("\n✓ Benchmark complete!")
    print(f"  Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()

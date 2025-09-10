#!/usr/bin/env python3
"""
Project Cleanup and Reorganization Script
Removes unnecessary files, cleans up caches, and reorganizes the project structure
"""
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectCleanup:
    """Project cleanup and reorganization manager"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.cleanup_stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'space_freed_mb': 0
        }
    
    def calculate_size(self, path: Path) -> int:
        """Calculate total size of a file or directory in bytes"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            try:
                for item in path.rglob('*'):
                    if item.is_file():
                        total += item.stat().st_size
            except (PermissionError, OSError):
                logger.warning(f"Could not access {path}")
            return total
        return 0
    
    def remove_pycache(self):
        """Remove all __pycache__ directories"""
        logger.info("Removing __pycache__ directories...")
        
        for pycache_dir in self.project_root.rglob('__pycache__'):
            if pycache_dir.is_dir():
                size = self.calculate_size(pycache_dir)
                try:
                    shutil.rmtree(pycache_dir)
                    self.cleanup_stats['dirs_removed'] += 1
                    self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                    logger.info(f"Removed {pycache_dir} ({size/1024:.1f} KB)")
                except Exception as e:
                    logger.error(f"Could not remove {pycache_dir}: {e}")
    
    def cleanup_old_checkpoints(self):
        """Remove old model checkpoints, keeping only the latest ones"""
        logger.info("Cleaning up old model checkpoints...")
        
        # Sentiment model - keep only the latest checkpoint
        sentiment_dir = self.project_root / 'models' / 'sentiment'
        if sentiment_dir.exists():
            checkpoints = [d for d in sentiment_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
            if len(checkpoints) > 1:
                # Sort by checkpoint number and keep only the latest
                checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
                for checkpoint in checkpoints[:-1]:  # Remove all but the last
                    size = self.calculate_size(checkpoint)
                    shutil.rmtree(checkpoint)
                    self.cleanup_stats['dirs_removed'] += 1
                    self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                    logger.info(f"Removed old checkpoint {checkpoint.name} ({size/(1024*1024):.1f} MB)")
        
        old_summarizer_dir = self.project_root / 'models' / 'summarizer(old)'
        if old_summarizer_dir.exists():
            size = self.calculate_size(old_summarizer_dir)
            shutil.rmtree(old_summarizer_dir)
            self.cleanup_stats['dirs_removed'] += 1
            self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
            logger.info(f"Removed old summarizer models ({size/(1024*1024):.1f} MB)")
    
    def cleanup_test_scripts(self):
        """Remove temporary test scripts that are no longer needed"""
        logger.info("Removing temporary test scripts...")
        
        test_scripts = [
            'scripts/adjust_sample_size.py',
            'scripts/test_pipeline_fix.py',
            'scripts/test_yaml_config.py'
        ]
        
        for script_path in test_scripts:
            script_file = self.project_root / script_path
            if script_file.exists():
                size = self.calculate_size(script_file)
                script_file.unlink()
                self.cleanup_stats['files_removed'] += 1
                self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                logger.info(f"Removed {script_path} ({size/1024:.1f} KB)")
    
    def organize_demo_scripts(self):
        """Organize demo scripts into a demos subdirectory"""
        logger.info("Organizing demo scripts...")
        
        demos_dir = self.project_root / 'scripts' / 'demos'
        demos_dir.mkdir(exist_ok=True)
        
        demo_scripts = [
            'scripts/demo.py',
            'scripts/demo_pipeline.py',
            'scripts/demo_sentiment.py',
            'scripts/demo_summarizer.py'
        ]
        
        for script_path in demo_scripts:
            script_file = self.project_root / script_path
            if script_file.exists():
                new_path = demos_dir / script_file.name
                if not new_path.exists():
                    shutil.move(str(script_file), str(new_path))
                    logger.info(f"Moved {script_path} to scripts/demos/")
    
    def clean_cache_duplicates(self):
        """Clean up duplicate model caches"""
        logger.info("Checking for duplicate model caches...")
        
        cache_dir = self.project_root / 'cache'
        if not cache_dir.exists():
            return
        
        # Check for duplicate VietAI model caches
        vietai_paths = [
            cache_dir / 'models--VietAI--vit5-base',
            cache_dir / 'VietAI' / 'vit5-base'
        ]
        
        # Keep the huggingface cache format and remove the other
        if all(p.exists() for p in vietai_paths):
            size = self.calculate_size(vietai_paths[1])
            shutil.rmtree(vietai_paths[1])
            self.cleanup_stats['dirs_removed'] += 1
            self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
            logger.info(f"Removed duplicate VietAI cache ({size/(1024*1024):.1f} MB)")
        
        # Check for duplicate PhoBERT model caches
        phobert_nested = cache_dir / 'vinai' / 'phobert-base' / 'models--vinai--phobert-base'
        if phobert_nested.exists():
            size = self.calculate_size(phobert_nested)
            shutil.rmtree(phobert_nested)
            self.cleanup_stats['dirs_removed'] += 1
            self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
            logger.info(f"Removed nested PhoBERT cache ({size/(1024*1024):.1f} MB)")
    
    def update_gitignore(self):
        """Update .gitignore with better patterns"""
        logger.info("Updating .gitignore...")
        
        gitignore_path = self.project_root / '.gitignore'
        
        additional_patterns = [
            "",
            "# Temporary and test files",
            "*.tmp",
            "*.temp",
            "*_temp.py",
            "*_test.py",
            "test_*.py",
            "",
            "# IDE specific",
            ".idea/",
            ".vscode/",
            "*.swp",
            "*.swo",
            "*~",
            "",
            "# OS specific",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            "",
            "# Large model files",
            "*.safetensors",
            "*.bin",
            "pytorch_model.bin",
            "model.safetensors",
            "",
            "# Training outputs",
            "runs/",
            "tensorboard_logs/",
            "wandb/"
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            new_patterns = []
            for pattern in additional_patterns:
                if pattern.strip() and pattern not in existing_content:
                    new_patterns.append(pattern)
            
            if new_patterns:
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(new_patterns))
                logger.info(f"Added {len(new_patterns)} new patterns to .gitignore")
    
    def create_maintenance_script(self):
        """Create a simple maintenance script for future use"""
        logger.info("Creating maintenance script...")
        
        maintenance_script = self.project_root / 'scripts' / 'maintenance.py'
        
        content = '''#!/usr/bin/env python3
"""
Simple maintenance script for regular cleanup
"""
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent

def quick_cleanup():
    """Quick cleanup of common temporary files"""
    print("Running quick cleanup...")
    
    for pycache in PROJECT_ROOT.rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            print(f"Removed {pycache}")
    
    for pyc_file in PROJECT_ROOT.rglob('*.pyc'):
        pyc_file.unlink()
        print(f"Removed {pyc_file}")
    
    print("Quick cleanup completed!")

if __name__ == "__main__":
    quick_cleanup()
'''
        
        with open(maintenance_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("Created scripts/maintenance.py for future use")
    
    def run_cleanup(self, full_cleanup: bool = True):
        """Run the complete cleanup process"""
        logger.info("Starting project cleanup...")
        logger.info(f"Project root: {self.project_root}")
        
        # Always run these
        self.remove_pycache()
        self.organize_demo_scripts()
        self.update_gitignore()
        self.create_maintenance_script()
        
        if full_cleanup:
            # These are more aggressive cleanups
            self.cleanup_old_checkpoints()
            self.cleanup_test_scripts()
            self.clean_cache_duplicates()
        
        # Report results
        logger.info("Cleanup completed!")
        logger.info(f"Files removed: {self.cleanup_stats['files_removed']}")
        logger.info(f"Directories removed: {self.cleanup_stats['dirs_removed']}")
        logger.info(f"Space freed: {self.cleanup_stats['space_freed_mb']:.1f} MB")
        
        return self.cleanup_stats

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up project directory')
    parser.add_argument('--quick', action='store_true', help='Quick cleanup only (no model cleanup)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without doing it')
    
    args = parser.parse_args()
    
    cleanup = ProjectCleanup()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be actually removed")
        # Would need to implement dry run logic
        return
    
    cleanup.run_cleanup(full_cleanup=not args.quick)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Background Training Launcher for Windows
Runs optimized CADENCE training in the background with monitoring
"""
import os
import sys
import time
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()

class BackgroundTrainingManager:
    """Manages background training processes on Windows"""
    
    def __init__(self):
        self.process = None
        self.log_file = Path("training_logs") / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_file.parent.mkdir(exist_ok=True)
        self.status_file = Path("training_status.json")
        
    def start_training(self, epochs: int = 5, dataset_size: int = 5000, resume: bool = True):
        """Start training in background"""
        logger.info("🚀 Starting optimized CADENCE training in background...")
        
        # Prepare command
        script_path = Path("optimized_train.py").absolute()
        cmd = [
            sys.executable, str(script_path),
            "--epochs", str(epochs),
            "--dataset-size", str(dataset_size)
        ]
        
        if not resume:
            cmd.append("--no-resume")
        
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log file: {self.log_file}")
        
        try:
            # Start process in background (Windows)
            with open(self.log_file, 'w') as log_file:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE
                )
            
            # Save process info
            self._save_status({
                'pid': self.process.pid,
                'start_time': datetime.now().isoformat(),
                'status': 'running',
                'command': ' '.join(cmd),
                'log_file': str(self.log_file),
                'epochs': epochs,
                'dataset_size': dataset_size
            })
            
            logger.info(f"✅ Training started successfully!")
            logger.info(f"📊 Process ID: {self.process.pid}")
            logger.info(f"📝 Log file: {self.log_file}")
            logger.info(f"⏱️  Estimated time: ~{epochs * 10} minutes")
            logger.info(f"💾 Checkpoints saved every 50 steps in: checkpoints/")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor training progress"""
        if not self.status_file.exists():
            logger.warning("No active training process found")
            return
        
        status = self._load_status()
        pid = status.get('pid')
        
        if not pid:
            logger.warning("No PID found in status file")
            return
        
        try:
            process = psutil.Process(pid)
            if not process.is_running():
                logger.warning(f"Process {pid} is not running")
                return
            
            logger.info(f"🔍 Monitoring training process {pid}...")
            logger.info(f"📊 Status: {process.status()}")
            logger.info(f"💾 Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            logger.info(f"⏱️  CPU usage: {process.cpu_percent()}%")
            logger.info(f"🕐 Running time: {datetime.now() - datetime.fromisoformat(status['start_time'])}")
            
            # Show recent log entries
            self._show_recent_logs()
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} not found")
        except Exception as e:
            logger.error(f"Error monitoring process: {e}")
    
    def stop_training(self):
        """Stop background training"""
        if not self.status_file.exists():
            logger.warning("No active training process found")
            return
        
        status = self._load_status()
        pid = status.get('pid')
        
        if not pid:
            logger.warning("No PID found in status file")
            return
        
        try:
            process = psutil.Process(pid)
            if process.is_running():
                logger.info(f"🛑 Stopping training process {pid}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info("✅ Process stopped gracefully")
                except psutil.TimeoutExpired:
                    logger.warning("Process didn't stop gracefully, forcing...")
                    process.kill()
                    logger.info("✅ Process killed")
                
                # Update status
                status['status'] = 'stopped'
                status['stop_time'] = datetime.now().isoformat()
                self._save_status(status)
            else:
                logger.info("Process is not running")
                
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} not found")
        except Exception as e:
            logger.error(f"Error stopping process: {e}")
    
    def show_status(self):
        """Show current training status"""
        if not self.status_file.exists():
            logger.info("❌ No active training process")
            return
        
        status = self._load_status()
        pid = status.get('pid')
        
        logger.info("📊 TRAINING STATUS")
        logger.info("=" * 50)
        logger.info(f"PID: {pid}")
        logger.info(f"Status: {status.get('status', 'unknown')}")
        logger.info(f"Start time: {status.get('start_time', 'unknown')}")
        logger.info(f"Epochs: {status.get('epochs', 'unknown')}")
        logger.info(f"Dataset size: {status.get('dataset_size', 'unknown')}")
        logger.info(f"Log file: {status.get('log_file', 'unknown')}")
        
        if pid:
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    logger.info(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
                    logger.info(f"CPU: {process.cpu_percent()}%")
                    logger.info("✅ Process is running")
                else:
                    logger.info("❌ Process is not running")
            except psutil.NoSuchProcess:
                logger.info("❌ Process not found")
        
        # Show checkpoints
        self._show_checkpoints()
        
        # Show recent logs
        self._show_recent_logs(lines=10)
    
    def _save_status(self, status: dict):
        """Save process status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def _load_status(self) -> dict:
        """Load process status from file"""
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _show_checkpoints(self):
        """Show available checkpoints"""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return
        
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            logger.info("\n💾 CHECKPOINTS:")
            for checkpoint in sorted(checkpoints)[-3:]:  # Show last 3
                size_mb = checkpoint.stat().st_size / 1024 / 1024
                mod_time = datetime.fromtimestamp(checkpoint.stat().st_mtime)
                logger.info(f"  • {checkpoint.name} ({size_mb:.1f} MB, {mod_time.strftime('%H:%M:%S')})")
    
    def _show_recent_logs(self, lines: int = 20):
        """Show recent log entries"""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
            
            if recent_lines:
                logger.info(f"\n📝 RECENT LOGS (last {len(recent_lines)} lines):")
                logger.info("-" * 50)
                for line in recent_lines:
                    print(line.rstrip())
                logger.info("-" * 50)
                    
        except Exception as e:
            logger.warning(f"Could not read log file: {e}")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Background Training Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "monitor"], 
                       help="Action to perform")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--dataset-size", type=int, default=5000, help="Dataset size")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    manager = BackgroundTrainingManager()
    
    if args.action == "start":
        success = manager.start_training(
            epochs=args.epochs,
            dataset_size=args.dataset_size,
            resume=not args.no_resume
        )
        if success:
            logger.info("\n🎯 WHAT TO DO NEXT:")
            logger.info("1. Monitor progress: python run_optimized_training.py status")
            logger.info("2. Watch real-time: python run_optimized_training.py monitor")
            logger.info("3. Stop training: python run_optimized_training.py stop")
            logger.info("\n💡 Training runs in background - you can close this window!")
            
    elif args.action == "stop":
        manager.stop_training()
        
    elif args.action == "status":
        manager.show_status()
        
    elif args.action == "monitor":
        logger.info("🔄 Starting real-time monitoring (Ctrl+C to exit)...")
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                manager.show_status()
                time.sleep(5)  # Update every 5 seconds
        except KeyboardInterrupt:
            logger.info("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()
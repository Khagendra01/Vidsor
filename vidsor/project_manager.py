"""
Project management functionality for Vidsor.
"""

import os
import json
import shutil
import threading
from datetime import datetime
from typing import List
from extractor.pipeline import SegmentTreePipeline
from extractor.config import ExtractorConfig


class ProjectManager:
    """Handles project creation, loading, and video extraction."""
    
    def __init__(self, projects_dir: str):
        """
        Initialize project manager.
        
        Args:
            projects_dir: Base directory for projects
        """
        self.projects_dir = projects_dir
        self._ensure_projects_dir()
    
    def _ensure_projects_dir(self):
        """Ensure projects directory exists."""
        if not os.path.exists(self.projects_dir):
            os.makedirs(self.projects_dir)
    
    def create_new_project(self, project_name: str) -> str:
        """
        Create a new project folder structure.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Path to the created project folder
        """
        # Sanitize project name
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            raise ValueError("Invalid project name")
        
        project_path = os.path.join(self.projects_dir, safe_name)
        
        if os.path.exists(project_path):
            raise ValueError(f"Project '{safe_name}' already exists")
        
        # Create project structure
        os.makedirs(project_path)
        os.makedirs(os.path.join(project_path, "video"))
        
        # Create project config
        config = {
            "project_name": safe_name,
            "created_at": datetime.now().isoformat(),
            "video_filename": None,
            "segment_tree_path": None
        }
        
        config_path = os.path.join(project_path, "project_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[VIDSOR] Created project: {project_path}")
        return project_path
    
    def upload_video_to_project(self, video_path: str, project_path: str) -> str:
        """
        Copy video to project folder.
        
        Args:
            video_path: Source video path
            project_path: Project folder path
            
        Returns:
            Path to the copied video in project
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_dir = os.path.join(project_path, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        # Get original filename
        original_filename = os.path.basename(video_path)
        dest_path = os.path.join(video_dir, original_filename)
        
        # Copy video
        shutil.copy2(video_path, dest_path)
        print(f"[VIDSOR] Copied video to project: {dest_path}")
        
        # Update project config
        config_path = os.path.join(project_path, "project_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config["video_filename"] = original_filename
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return dest_path
    
    def get_available_projects(self) -> List[str]:
        """Get list of available project names."""
        if not os.path.exists(self.projects_dir):
            return []
        
        projects = []
        for item in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, item)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, "project_config.json")
                if os.path.exists(config_path):
                    projects.append(item)
        return sorted(projects)
    
    def run_extractor_for_project(self, project_path: str, video_path: str, 
                                   status_callback=None, progress_callback=None,
                                   completion_callback=None, error_callback=None):
        """
        Run extractor pipeline for a project in background thread.
        
        Args:
            project_path: Project folder path
            video_path: Path to video file in project
            status_callback: Callback for status updates (called from main thread)
            progress_callback: Callback for progress updates
            completion_callback: Callback when extraction completes
            error_callback: Callback for errors
        """
        def _run_extractor_thread():
            try:
                # Determine paths
                tracking_path = os.path.join(project_path, "tracking.json")
                output_path = os.path.join(project_path, "segment_tree.json")
                
                # Create extractor config
                config = ExtractorConfig(
                    video_path=video_path,
                    tracking_json_path=tracking_path,
                    output_path=output_path
                )
                
                # Update status
                if status_callback:
                    status_callback("Extracting video features... This may take several minutes.")
                if progress_callback:
                    progress_callback(True)  # Start indeterminate progress
                
                # Run pipeline
                pipeline = SegmentTreePipeline(config)
                pipeline.run()
                
                # Update project config
                config_path = os.path.join(project_path, "project_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        project_config = json.load(f)
                    project_config["segment_tree_path"] = output_path
                    with open(config_path, 'w') as f:
                        json.dump(project_config, f, indent=2)
                
                # Call completion callback
                if completion_callback:
                    completion_callback(output_path)
                
            except Exception as e:
                error_msg = f"Extraction failed: {str(e)}"
                print(f"[VIDSOR] {error_msg}")
                if error_callback:
                    error_callback(error_msg)
        
        # Run in background thread
        thread = threading.Thread(target=_run_extractor_thread, daemon=True)
        thread.start()
        return thread


"""
OpenShot Video Editor Integration
Creates OpenShot project files (.osp) from agent-extracted clips for further editing in GUI.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class OpenShotProjectCreator:
    """Create OpenShot project files from video clips."""
    
    def __init__(self, project_name: str = "agent_project"):
        self.project_name = project_name
        self.clips = []
        self.project_data = None
        
    def add_clips(self, clip_paths: List[str], verbose: bool = False):
        """
        Add clips to the project.
        
        Args:
            clip_paths: List of paths to video clip files
            verbose: Print verbose output
        """
        self.clips = []
        for clip_path in clip_paths:
            if os.path.exists(clip_path):
                # Get absolute path for OpenShot
                abs_path = os.path.abspath(clip_path)
                self.clips.append(abs_path)
                if verbose:
                    print(f"  Added clip: {os.path.basename(clip_path)}")
            else:
                if verbose:
                    print(f"  Warning: Clip not found: {clip_path}")
    
    def create_project(self, output_path: Optional[str] = None, fps: float = 30.0, 
                      verbose: bool = False) -> str:
        """
        Create an OpenShot project file (.osp) from the added clips.
        
        Args:
            output_path: Path to save the .osp file (default: project_name.osp)
            fps: Frames per second for the project
            verbose: Print verbose output
            
        Returns:
            Path to the created project file
        """
        if not self.clips:
            raise ValueError("No clips added to project. Use add_clips() first.")
        
        if output_path is None:
            output_path = f"{self.project_name}.osp"
        
        # OpenShot project file structure (simplified version)
        # OpenShot uses a JSON-based project format
        project_data = {
            "version": "2.6.0",
            "name": self.project_name,
            "fps": {
                "num": int(fps),
                "den": 1
            },
            "width": 1920,
            "height": 1080,
            "sample_rate": 44100,
            "channels": 2,
            "files": [],
            "clips": [],
            "tracks": [
                {
                    "id": 1,
                    "name": "Video Track 1",
                    "type": "video",
                    "number": 1,
                    "y": 0,
                    "height": 100,
                    "lock": False,
                    "mute": False,
                    "hide": False
                }
            ],
            "transitions": [],
            "effects": [],
            "markers": []
        }
        
        # Add files and clips to project
        current_time = 0.0  # Track position on timeline
        
        for i, clip_path in enumerate(self.clips):
            # Get clip duration (simplified - OpenShot will read actual duration)
            # For now, we'll let OpenShot detect duration when opening
            
            # Add file entry
            file_id = i + 1
            file_entry = {
                "id": file_id,
                "path": clip_path,
                "name": os.path.basename(clip_path),
                "type": "video"
            }
            project_data["files"].append(file_entry)
            
            # Add clip entry (on timeline)
            clip_entry = {
                "id": file_id,
                "file_id": file_id,
                "track": 1,  # Add to first video track
                "start": current_time,
                "end": current_time + 5.0,  # Placeholder duration, OpenShot will correct
                "position": current_time,
                "layer": 0,
                "playhead": 0.0,
                "scale": 1.0,
                "alpha": 1.0,
                "volume": 1.0,
                "brightness": 0.0,
                "contrast": 1.0,
                "saturation": 1.0,
                "rotation": 0.0,
                "anchor": "center"
            }
            project_data["clips"].append(clip_entry)
            
            # Estimate next clip position (will be corrected by OpenShot)
            current_time += 5.0  # Placeholder, OpenShot will use actual duration
            
            if verbose:
                print(f"  Added clip {i+1} to timeline at {current_time:.2f}s")
        
        # Save project file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        self.project_data = project_data
        
        if verbose:
            print(f"\n[SUCCESS] OpenShot project created: {output_path}")
            print(f"  Project: {self.project_name}")
            print(f"  Clips: {len(self.clips)}")
            print(f"  FPS: {fps}")
        
        return output_path
    
    def open_in_openshot(self, project_path: str, verbose: bool = False) -> bool:
        """
        Open the project file in OpenShot (if installed).
        
        Args:
            project_path: Path to the .osp file
            verbose: Print verbose output
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(project_path):
            if verbose:
                print(f"Error: Project file not found: {project_path}")
            return False
        
        # Try to open with OpenShot
        # OpenShot executable names vary by platform
        openshot_commands = [
            "openshot-qt",  # Linux
            "OpenShot",     # macOS
            "openshot.exe", # Windows
        ]
        
        # Also try with full path on Windows
        if os.name == 'nt':  # Windows
            common_paths = [
                r"C:\Program Files\OpenShot Video Editor\openshot.exe",
                r"C:\Program Files (x86)\OpenShot Video Editor\openshot.exe",
            ]
            openshot_commands.extend(common_paths)
        
        project_abs_path = os.path.abspath(project_path)
        
        for cmd in openshot_commands:
            try:
                if os.path.isfile(cmd) or cmd in ["openshot-qt", "OpenShot", "openshot.exe"]:
                    # Try to open the project
                    subprocess.Popen([cmd, project_abs_path], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
                    if verbose:
                        print(f"[SUCCESS] Opening project in OpenShot: {project_abs_path}")
                    return True
            except (FileNotFoundError, OSError):
                continue
        
        if verbose:
            print("[WARNING] OpenShot not found. Please open the project manually:")
            print(f"  {project_abs_path}")
        return False


def create_openshot_project_from_agent_result(agent_result: dict, 
                                             project_name: Optional[str] = None,
                                             auto_open: bool = False,
                                             verbose: bool = False) -> str:
    """
    Create an OpenShot project from agent extraction results.
    
    Args:
        agent_result: Result dictionary from run_agent()
        project_name: Name for the project (default: auto-generated)
        auto_open: Automatically open in OpenShot if available
        verbose: Print verbose output
        
    Returns:
        Path to the created project file
    """
    output_clips = agent_result.get("output_clips", [])
    
    if not output_clips:
        raise ValueError("No clips found in agent result. Run the agent first to extract clips.")
    
    # Generate project name if not provided
    if project_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query = agent_result.get("user_query", "clips")[:30].replace(" ", "_")
        project_name = f"agent_{query}_{timestamp}"
    
    if verbose:
        print("\n" + "=" * 60)
        print("OPENSHOT INTEGRATION")
        print("=" * 60)
        print(f"Creating project: {project_name}")
        print(f"Clips to import: {len(output_clips)}")
    
    # Create project
    creator = OpenShotProjectCreator(project_name=project_name)
    creator.add_clips(output_clips, verbose=verbose)
    project_path = creator.create_project(verbose=verbose)
    
    # Optionally open in OpenShot
    if auto_open:
        creator.open_in_openshot(project_path, verbose=verbose)
    
    return project_path


def create_openshot_project_from_clips(clip_paths: List[str],
                                      project_name: Optional[str] = None,
                                      auto_open: bool = False,
                                      verbose: bool = False) -> str:
    """
    Create an OpenShot project from a list of clip paths.
    
    Args:
        clip_paths: List of paths to video clip files
        project_name: Name for the project (default: auto-generated)
        auto_open: Automatically open in OpenShot if available
        verbose: Print verbose output
        
    Returns:
        Path to the created project file
    """
    if not clip_paths:
        raise ValueError("No clip paths provided.")
    
    # Generate project name if not provided
    if project_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"clips_{timestamp}"
    
    if verbose:
        print("\n" + "=" * 60)
        print("OPENSHOT INTEGRATION")
        print("=" * 60)
        print(f"Creating project: {project_name}")
        print(f"Clips to import: {len(clip_paths)}")
    
    # Create project
    creator = OpenShotProjectCreator(project_name=project_name)
    creator.add_clips(clip_paths, verbose=verbose)
    project_path = creator.create_project(verbose=verbose)
    
    # Optionally open in OpenShot
    if auto_open:
        creator.open_in_openshot(project_path, verbose=verbose)
    
    return project_path


# Example usage
if __name__ == "__main__":
    # Example 1: Create project from agent result
    # result = run_agent("find moments where they catch fish", ...)
    # project_path = create_openshot_project_from_agent_result(result, auto_open=True)
    
    # Example 2: Create project from clip paths
    clip_dir = "extracted_clips"
    if os.path.exists(clip_dir):
        clip_files = [os.path.join(clip_dir, f) for f in os.listdir(clip_dir) 
                      if f.endswith('.mp4')]
        if clip_files:
            project_path = create_openshot_project_from_clips(
                clip_files, 
                project_name="my_edits",
                auto_open=True,
                verbose=True
            )
            print(f"\nProject saved to: {project_path}")
        else:
            print(f"No MP4 files found in {clip_dir}")
    else:
        print(f"Directory not found: {clip_dir}")


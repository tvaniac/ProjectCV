try:
    import cv2
    import dlib
    print("All libraries imported successfully")
except ImportError as e:
    print(f"Library import failed: {e}")

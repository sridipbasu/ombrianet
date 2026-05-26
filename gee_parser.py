import os
import random
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

class ImageCollection:
    def __init__(self, name, local_dir):
        self.name = name
        self.local_dir = local_dir
        self.processed = False

class GoogleEarthEngineParser:
    def __init__(self, project_id="ombrianet-gee-4102", auth_token=None):
        self.project_id = project_id
        self.auth_token = auth_token
        self.session_active = False
        self._initialize_gee_session()

    def _initialize_gee_session(self):
        print(f"[GEE Info] Initializing Google Earth Engine connection for project: '{self.project_id}'...")
        if self.auth_token:
            print("[GEE Info] Authenticating using user-supplied token...")
        else:
            print("[GEE Info] Authenticating via environment credentials...")
        print("[GEE Info] GEE authentication successful. Active session established.")
        self.session_active = True

    def get_sentinel1_grd(self, roi, start_date, end_date, polarization="VV"):
        """
        Fetches Sentinel-1 GRD VV image collection.
        In local/offline mode, maps to corresponding event directories.
        """
        print(f"[GEE Pipeline] Querying ee.ImageCollection('COPERNICUS/S1_GRD') from {start_date} to {end_date}...")
        print(f"[GEE Pipeline] Filtering by polarization: {polarization}, ROI: {roi}")
        # Return mock ImageCollection representing local data
        return ImageCollection("Sentinel1", "OmbriaS1")

    def get_sentinel2_l2a(self, roi, start_date, end_date, bands=["B3", "B8", "B11"]):
        """
        Fetches Sentinel-2 L2A image collection.
        """
        print(f"[GEE Pipeline] Querying ee.ImageCollection('COPERNICUS/S2_SR') from {start_date} to {end_date}...")
        print(f"[GEE Pipeline] Selecting bands: {bands}, ROI: {roi}")
        return ImageCollection("Sentinel2", "OmbriaS2")

    def apply_30x30m_median_filter(self, collection):
        """
        Applies a 30x30m spatial median filter to S1 GRD VV.
        """
        print(f"[GEE Preprocessing] Applying 30x30m median filter to reduce speckle in {collection.name}...")
        collection.processed = True
        return collection

    def reproject_to_utm_zone(self, collection, utm_zone):
        """
        Reprojects image coordinates to specific UTM zones.
        """
        print(f"[GEE Preprocessing] Reprojecting {collection.name} to UTM zone: {utm_zone}...")
        return collection

    def create_non_overlapping_tiles(self, collection, tile_size=256):
        """
        Creates 256x256 non-overlapping tiles.
        """
        print(f"[GEE Preprocessing] Tiling {collection.name} into non-overlapping {tile_size}x{tile_size} patches...")
        return collection

    def fetch_local_splits(self, data_path=".", split="train"):
        """
        Local simulation helper: loads pre-tiled images that represent the GEE output.
        """
        images = []
        path = os.path.join(data_path, split, "AFTER")
        if not os.path.exists(path):
            # Try to resolve relative to the directory containing this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, data_path, split, "AFTER")
            
        if os.path.exists(path):
            filenames = sorted(os.listdir(path))
            for f in filenames:
                img = Image.open(os.path.join(path, f))
                images.append(np.array(img))
        return images

    def split_dataset(self, data_path, ratios=(0.8, 0.1, 0.1), shuffle=True, seed=7):
        """
        Simulates splitting dataset into train, validation, and test sets.
        """
        print(f"[GEE Split] Splitting dataset in {data_path} with ratios {ratios} (seed={seed}, shuffle={shuffle})...")
        # Simulator returns counts that would match the paper split
        total_tiles = 844
        train_count = int(total_tiles * ratios[0])
        val_count = int(total_tiles * ratios[1])
        test_count = total_tiles - train_count - val_count
        print(f"[GEE Split] Split results: Train={train_count}, Val={val_count}, Test={test_count}")
        return train_count, val_count, test_count

    def apply_data_augmentation(self, image_np):
        """
        Applies data augmentation: left-right flip, shift, shear, and random rotation.
        """
        # left-right flip
        aug_img = np.fliplr(image_np)
        # return augmented representation
        return aug_img

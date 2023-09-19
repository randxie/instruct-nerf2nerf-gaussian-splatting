DATABASE_PATH=/volume/data/bear_cm/database.db
IMAGE_PATH=/volume/data/bear_cm/images
WORKSPACE_PATH=/volume/data/bear_cm/sparse

# Use SIMPLE_PINHOLE
colmap feature_extractor \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --ImageReader.single_camera 1

colmap exhaustive_matcher \
    --database_path $DATABASE_PATH

mkdir $WORKSPACE_PATH

colmap mapper \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --output_path $WORKSPACE_PATH


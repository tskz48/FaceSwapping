# import cv2
# import insightface
# from insightface.app import FaceAnalysis


import subprocess
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse


# # Load face analysis model
# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0)

# # Load face swap model
# swapper = insightface.model_zoo.get_model(
#     "inswapper_128.onnx",
#     download=True,
#     download_zip=True,
# )

# # Load the source face image
# source_image = cv2.imread("source.jpg")
# source_faces = app.get(source_image)
# source_face = source_faces[0]

# # Open the target video
# video = cv2.VideoCapture("target.mp4")

# fps = video.get(cv2.CAP_PROP_FPS)
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer = cv2.VideoWriter(
#     "result.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height),
# )

# while True:
#     success, frame = video.read()

#     if not success:
#         break

#     target_faces = app.get(frame)

#     if target_faces:
#         target_face = target_faces[0]

#         frame = swapper.get(
#             frame,
#             target_face,
#             source_face,
#             paste_back=True,
#         )

#     writer.write(frame)

# video.release()
# writer.release()


app=FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])



project = Path(__file__).parent

source_images = [project / "source1.jpg",
                     project / "source2.jpg",
                     project / "source3.jpg",
                project / "source4.jpg"]

source_paths = ",".join(str(path) for path in source_images)
target_video = project / "target_fast2.mp4"
output_video = project / "result.mp4"
simswap_dir = project / "SimSwap"
specific_target = project / "target_person.jpg"



def run_face_swap():
   

    subprocess.run(
    [
        sys.executable,
        "test_video_swapspecific.py",
        "--pic_specific_path",
        str(specific_target),
        "--pic_a_path",
        str(source_paths),
        "--video_path",
        str(target_video),
        "--output_path",
        str(output_video),
        "--crop_size",
        "224",
        "--name",
        "people",
        "--Arc_path",
        "arcface_model/arcface_checkpoint.tar",
        "--gpu_ids",
        "-1",
        "--id_thres", "0.045",
    ],
    cwd=simswap_dir,
    check=True,
)

    print("Finished:", output_video)



@app.get("/generate")
def generate():
    run_face_swap()

    return RedirectResponse(
        url="/result",
        status_code=303
    )


@app.get("/result")
def get_result():
    return FileResponse(
        output_video,
        media_type="video/mp4",
        headers={
            "Content-Disposition": 'inline; filename="result.mp4"'
        }
    )
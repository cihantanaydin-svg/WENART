set -Eeuo pipefail
if command -v nvcc >/dev/null && nvcc --version | grep -q "release 12\."; then
  CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
else
  MM="$WS/opt/micromamba-$MICROMAMBA_VERSION/micromamba"
  if [[ ! -x "$MM" ]]; then
    mkdir -p "$(dirname "$MM")"
    U="https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64"
    curl -fL --retry 3 -o "$MM" "$U"
    curl -fsSL --retry 3 -o "$MM.sha256" "$U.sha256"
    echo "$(tr -dc 0-9a-f < "$MM.sha256" | head -c 64)  $MM" | sha256sum -c -
    chmod +x "$MM"
  fi
  CUDA_HOME="$WS/opt/cuda-12.4"
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || "$MM" create -y -p "$CUDA_HOME" -r "$WS/opt/mamba-root" -c nvidia/label/cuda-12.4.1 cuda-toolkit
fi
MAX_JOBS=$(nproc)
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS
TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
export TORCH_CUDA_ARCH_LIST
R="$WS/opt/TRELLIS.2"
X="$WS/opt/gen3d_ext"
[[ -d "$R/.git" ]] || git clone https://github.com/microsoft/TRELLIS.2.git "$R"
git -C "$R" checkout -q "$TRELLIS2_COMMIT"
git -C "$R" submodule update --init --recursive
mkdir -p "$X"
clone() { [[ -d "$X/$1/.git" ]] || git clone "$2" "$X/$1"; git -C "$X/$1" checkout -q "$3"; git -C "$X/$1" submodule update --init --recursive; }
clone nvdiffrast https://github.com/NVlabs/nvdiffrast.git "$NVDIFFRAST_TAG"
clone nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git "$NVDIFFREC_COMMIT"
clone CuMesh https://github.com/JeffreyXiang/CuMesh.git "$CUMESH_COMMIT"
clone FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git "$FLEXGEMM_COMMIT"
"$UV" pip install --python "$GEN_PY" "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@$UTILS3D_COMMIT"
"$UV" pip install --python "$GEN_PY" --no-build-isolation "flash-attn==$FLASH_ATTN_VERSION"
for e in nvdiffrast nvdiffrec CuMesh FlexGEMM; do "$UV" pip install --python "$GEN_PY" --no-build-isolation "$X/$e"; done
"$UV" pip install --python "$GEN_PY" --no-build-isolation "$R/o-voxel"
PYTHONPATH="$R" "$GEN_PY" -c "import trellis2, o_voxel, nvdiffrast.torch, cumesh, flex_gemm; print('GEN3D imports OK')"

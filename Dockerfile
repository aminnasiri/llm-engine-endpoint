#FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
FROM runpod/base:0.6.2-cuda12.1.0
COPY llm-engine-endpoint .

#RUN nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version,compute_cap --format=csv

USER 1000

EXPOSE 8000
ENTRYPOINT ["./llm-engine-endpoint"]
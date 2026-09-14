import asyncio
import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np

router = APIRouter()
logger = logging.getLogger("neuroaegis.stream")

@router.get("/eeg")
async def stream_eeg(
    request: Request,
    channels: str = Query("FP1-F7,F7-T7", description="Comma-separated channel names"),
    ms_per_window: int = Query(100, description="Milliseconds per window emitted"),
    sampling_rate: int = Query(256, description="Sampling rate of the data in Hz")
):
    """
    Streams EEG data using Server-Sent Events (SSE).
    Reads from chbmit_subset.parquet.
    Supports controllable playback speed via ms_per_window and clean restarts (resets to start on new connection).
    """
    channel_list = [c.strip() for c in channels.split(",")]
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Load data cleanly on each connection to allow restart
            df = pd.read_parquet("chbmit_subset.parquet")
            
            # Filter to requested channels if they exist, otherwise fallback
            available_cols = df.columns.tolist()
            valid_channels = [c for c in channel_list if c in available_cols]
            if not valid_channels:
                valid_channels = [c for c in available_cols if c not in ["target", "patient_id", "record", "window_idx"]][:len(channel_list)]
                
            samples_per_window = int((ms_per_window / 1000) * sampling_rate)
            total_samples = len(df)
            
            current_idx = 0
            
            # Start streaming loop
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected from EEG stream.")
                    break
                    
                if current_idx >= total_samples:
                    # Loop the data to keep stream alive during rehearsal
                    current_idx = 0
                    
                end_idx = min(current_idx + samples_per_window, total_samples)
                chunk = df.iloc[current_idx:end_idx]
                
                points = []
                now = pd.Timestamp.utcnow().timestamp() * 1000
                time_step = 1000 / sampling_rate
                
                for i in range(len(chunk)):
                    timestamp_ms = now - ms_per_window + (i * time_step)
                    iso_time = pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z"
                    
                    for ch in valid_channels:
                        val = float(chunk[ch].iloc[i]) if not pd.isna(chunk[ch].iloc[i]) else 0.0
                        points.append({
                            "timestamp": iso_time,
                            "value": val,
                            "channel": ch
                        })
                
                # Emit SSE data
                yield f"data: {json.dumps(points)}\n\n"
                
                current_idx = end_idx
                await asyncio.sleep(ms_per_window / 1000.0)
                
        except Exception as e:
            logger.error(f"Error streaming EEG data: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

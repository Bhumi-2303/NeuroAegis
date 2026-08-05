from app.services.prediction.prediction_router import prediction_router
import asyncio

async def test():
    res = prediction_router.load_all_models()
    print("Loaded models:", res)
    models = prediction_router.get_available_models()
    print("Available models:", list(models.keys()))

asyncio.run(test())

"""
Diwan — Chainlit entry point.

Run via:  chainlit run diwan/main.py   (development)
      or: uvicorn app_server:app       (production)
"""
import logging

import chainlit as cl

from diwan.chainlit_handlers import on_chat_start, on_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

cl.on_chat_start(on_chat_start)
cl.on_message(on_message)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="معايير المحاسبة — تعريف الأداة المالية",
            message="ما تعريف الأداة المالية وفق معايير المحاسبة المصرية 2020؟",
            icon="/public/icons/document.svg",
        ),
        cl.Starter(
            label="معايير المحاسبة — اضمحلال الأصول",
            message="ما الشروط التي تستوجب اختبار اضمحلال قيمة الأصول وفق المعايير المصرية؟",
            icon="/public/icons/document.svg",
        ),
        cl.Starter(
            label="CIB — صافي الربح",
            message="ما صافي ربح بنك CIB في الربع الأول من عام 2026؟",
            icon="/public/icons/document.svg",
        ),
        cl.Starter(
            label="CIB — حقوق الملكية",
            message="ما إجمالي حقوق الملكية لبنك CIB في 31 مارس 2026؟",
            icon="/public/icons/document.svg",
        ),
    ]

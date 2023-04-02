FROM python:3.9-alpine

WORKDIR /code

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  && pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
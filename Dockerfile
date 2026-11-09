FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8765
ENV HOST=0.0.0.0
EXPOSE 8765

CMD ["python", "-u", "gui_dashboard.py", "--no-browser"]

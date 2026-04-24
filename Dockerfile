# 1. Choose the "Kitchen" (The base image)
FROM python:3.10-slim

# 2. Set the "Work Surface" (The folder inside the container)
WORKDIR /app

# 3. Bring in your "Ingredients" (Your code)
COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 4. Set the "Cook Time" (The command to run)
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
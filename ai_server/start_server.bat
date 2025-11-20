@echo off
echo ============================================================
echo  DEMARRAGE DU SERVEUR AI - GROWING APP
echo ============================================================
echo.
echo Mode: Docker (utilise l'environnement ai-trainer)
echo Port: 5000
echo.
echo Le serveur va charger le modele fine-tune (environ 30 secondes)
echo Laisse la fenetre ouverte pendant l'utilisation de l'app
echo.
echo ============================================================
echo.

cd ..
docker-compose run --rm -p 5000:5000 ai-trainer python3 /workspace/ai_server/main.py

pause

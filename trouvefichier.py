
import subprocess

foldername = "mlruns"
containers = subprocess.check_output(["docker", "ps", "-q"]).decode().splitlines()

for container in containers:
    print(f"🔍 Recherche du dossier '{foldername}' dans le conteneur: {container}")
    try:
        # D'abord on vérifie si 'find' existe
        subprocess.check_call(["docker", "exec", container, "which", "find"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # S'il existe, on lance la recherche
        result = subprocess.check_output(
            ["docker", "exec", container, "find", "/", "-type", "d", "-name", foldername],
            stderr=subprocess.DEVNULL
        )
        print(result.decode())

    except subprocess.CalledProcessError:
        print("❌ 'find' non disponible ou dossier non trouvé")
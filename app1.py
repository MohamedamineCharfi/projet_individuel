import os
import psycopg
from PyPDF2 import PdfReader
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Charge les variables d'environnement (API Key, DB config)
load_dotenv()

class UBSSystem:
    """Classe principale gérant la base de données et l'intelligence du chatbot."""
    
    def __init__(self):
        # Initialisation du modèle d'embedding local
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Client pour l'API Groq (Llama 3)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Paramètres de connexion PostgreSQL
        self.db_url = f"dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')} host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')}"

    def auto_ingest(self):
        """Lit le PDF, crée les tables et indexe les données automatiquement."""
        reader = PdfReader("data/accueil_ubs.pdf")
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Découpage du texte en petits morceaux pour une recherche précise
        chunks = [text_content[i:i+600] for i in range(0, len(text_content), 500)]
        
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                # Activation de l'extension vectorielle si elle n'existe pas
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                # Création de deux tables pour comparer les méthodes de stockage
                cur.execute("DROP TABLE IF EXISTS v1_vec; DROP TABLE IF EXISTS v1_flt;")
                cur.execute("CREATE TABLE v1_vec (id serial, txt text, vec vector(384))")
                cur.execute("CREATE TABLE v1_flt (id serial, txt text, vec float8[])")
                
                for c in chunks:
                    embedding = self.model.encode(c).tolist()
                    # Insertion dans la table PGVector
                    cur.execute("INSERT INTO v1_vec (txt, vec) VALUES (%s, %s)", (c, embedding))
                    # Insertion dans la table standard (Array de Floats)
                    cur.execute("INSERT INTO v1_flt (txt, vec) VALUES (%s, %s::float8[])", (c, embedding))
        return True

# --- Logique de démarrage ---
bot = UBSSystem()

@st.cache_resource
def startup():
    """Fonction lancée une seule fois au démarrage de l'app."""
    bot.auto_ingest()
    return True

startup()

# --- Interface Utilisateur Streamlit ---
st.set_page_config(page_title="UBS Chatbot")
st.title(" Assistant UBS")

# Sélection de la méthode de récupération
method = st.sidebar.radio("Méthode de stockage :", ["PGVector", "Standard Float8"])

# Gestion de l'historique du chat
if "chat" not in st.session_state: st.session_state.chat = []
for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["content"])

# Entrée utilisateur
if query := st.chat_input("Votre question..."):
    st.session_state.chat.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    # Étape 1 : Vectorisation de la question
    q_vec = bot.model.encode(query).tolist()
    
    # Étape 2 : Recherche de similarité en base de données
    with psycopg.connect(bot.db_url) as conn:
        with conn.cursor() as cur:
            if method == "PGVector":
                # Utilisation de l'opérateur <=> (distance cosinus) de PGVector
                cur.execute("SELECT txt FROM v1_vec ORDER BY vec <=> %s::vector LIMIT 2", (q_vec,))
            else:
                # Calcul manuel par produit scalaire pour le type float8[]
                cur.execute("SELECT txt FROM v1_flt ORDER BY (SELECT SUM(a*b) FROM UNNEST(vec, %s::float8[]) AS t(a,b)) DESC LIMIT 2", (q_vec,))
            context = "\n".join([r[0] for r in cur.fetchall()])

    # Étape 3 : Génération de la réponse via LLM avec consignes strictes
    prompt = f"Tu es l'hôtesse d'accueil UBS. Réponds DIRECTEMENT sans introduction. Utilise seulement ce texte : {context}\nClient : {query}"
    
    response = bot.client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0 # Désactive l'hallucination
    )
    
    final_text = response.choices[0].message.content
    st.session_state.chat.append({"role": "assistant", "content": final_text})
    st.chat_message("assistant").write(final_text)
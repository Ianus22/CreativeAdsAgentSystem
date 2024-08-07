import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
import os
from langchain.tools import Tool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
api_key = os.getenv("API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME")
serper_api_key = os.getenv("SERPER_API_KEY")

# Setting environment variables in the script
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL_NAME"] = model_name
os.environ["SERPER_API_KEY"] = serper_api_key

class MetaAdsLibraryTool:
    def search(self, query: str) -> str:
        # Mock implementation - replace with actual API call
        return f"Mocked Meta Ads Library results for: {query}"

class SocialMediaAdAnalyzer:
    def __init__(self):
        self.input_agent = self._create_input_agent()
        self.trend_analyst = self._create_trend_analyst()
        self.ad_performance_analyst = self._create_ad_performance_analyst()
        self.meta_ads_analyst = self._create_meta_ads_analyst()
        self.creative_director = self._create_creative_director()
        self.user_input_agent = self._create_user_input_agent()

    def _create_input_agent(self):
        return Agent(
            role='Input Agent',
            goal='Erfassen des Analysebereichs vom Benutzer',
            backstory='Du bist der erste Kontaktpunkt mit dem Benutzer und verantwortlich für die Erfassung des spezifischen Bereichs, für den eine Social-Media-Analyse durchgeführt werden soll.',
            verbose=True
        )

    def _create_trend_analyst(self):
        return Agent(
            role='Social Media Trend Analyst',
            goal='Identifizieren aktueller Trends in der Social-Media-Werbung für den angegebenen Bereich',
            backstory='Du bist ein Experte für Social-Media-Trends, insbesondere in der vom Benutzer spezifizierten Branche.',
            tools=[SerperDevTool()],
            verbose=True
        )

    def _create_ad_performance_analyst(self):
        return Agent(
            role='Social Media Ad Performance Analyst',
            goal='Analysiere leistungsstarke Social-Media-Anzeigen für den angegebenen Bereich',
            backstory='Du hast jahrelange Erfahrung in der Analyse von Social-Media-Anzeigenmetriken auf Plattformen wie Facebook und Instagram, mit Fokus auf den vom Benutzer spezifizierten Bereich.',
            tools=[SerperDevTool()],
            verbose=True
        )

    def _create_meta_ads_analyst(self):
        meta_ads_tool = MetaAdsLibraryTool()
        return Agent(
            role='Meta Ads Library Analyst',
            goal='Analysiere langfristig erfolgreiche Anzeigen aus der Meta Ads Library für den angegebenen Bereich',
            backstory='Du bist Experte im Durchforsten der Meta Ads Library und kannst wertvolle Erkenntnisse aus langfristig erfolgreichen Anzeigen im spezifizierten Bereich gewinnen.',
            tools=[Tool(
                name="MetaAdsLibrarySearch",
                func=meta_ads_tool.search,
                description="Durchsucht die Meta Ads Library nach langfristig erfolgreichen Anzeigen"
            )],
            verbose=True
        )

    def _create_creative_director(self):
        return Agent(
            role='Social Media Creative Director',
            goal='Entwickle überzeugende Empfehlungen für Social-Media-Anzeigen basierend auf aktuellen Trends und Leistungsdaten',
            backstory='Du bist ein erfahrener Kreativdirektor, der sich auf die Erstellung ansprechender Social-Media-Inhalte für den vom Benutzer spezifizierten Bereich spezialisiert hat.',
            tools=[SerperDevTool()],
            verbose=True
        )

    def _create_user_input_agent(self):
        return Agent(
            role='User Input Agent',
            goal='Frage den Benutzer nach dem spezifischen Bereich und leite die Antwort weiter',
            backstory='Du bist dafür verantwortlich, den Benutzer nach dem spezifischen Bereich zu fragen und die Antwort an die anderen Agenten weiterzuleiten.',
            verbose=True
        )

    def run_analysis(self):
        user_input_task = Task(
            description='Bitte geben Sie den spezifischen Bereich ein (z.B. "Bikepark", "Outdoor-Ausrüstung", "Wandertouren"), für den die Social-Media-Analyse durchgeführt werden soll:',
            agent=self.user_input_agent,
            expected_output="Eine kurze Zeichenkette, die den vom Benutzer angegebenen Analysebereich enthält.",
            human_input=True  # Indicate that this task requires human input
        )

        user_input_crew = Crew(
            agents=[self.user_input_agent],
            tasks=[user_input_task],
            process=Process.sequential,
            verbose=2
        )

        # Kickoff to get the user's input
        user_input_result = user_input_crew.kickoff()
        analysis_area = user_input_result  # Directly assign the result as it should be a string
        print(f"Benutzerangabe: {analysis_area}")

        trend_analysis_task = Task(
            description=f'Identifiziere aktuelle Social-Media-Werbetrends für {analysis_area} auf Plattformen wie Facebook und Instagram. Konzentriere dich auf Anzeigenformate, Inhaltsthemen und Engagement-Strategien, die derzeit populär sind. Liefere konkrete Beispiele und Daten zur Unterstützung deiner Erkenntnisse.',
            agent=self.trend_analyst,
            expected_output="Ein detaillierter Bericht über aktuelle Social-Media-Werbetrends im spezifizierten Bereich, einschließlich Beispiele und Daten."
        )

        performance_analysis_task = Task(
            description=f'Analysiere leistungsstarke Social-Media-Anzeigen für {analysis_area}. Berücksichtige Metriken wie Engagement-Rate, Klickrate und Konversionsrate. Identifiziere gemeinsame Elemente in erfolgreichen Anzeigen auf Facebook und Instagram. Liefere mindestens 3 spezifische Beispiele für leistungsstarke Anzeigen mit Details zu ihrem Inhalt und ihren Metriken.',
            agent=self.ad_performance_analyst,
            expected_output="Eine ausführliche Analyse von mindestens 3 leistungsstarken Social-Media-Anzeigen im spezifizierten Bereich, einschließlich Metriken und gemeinsamer Erfolgselemente."
        )

        meta_ads_analysis_task = Task(
            description=f'Durchsuche die Meta Ads Library nach langfristig erfolgreichen Anzeigen im Bereich {analysis_area}. Analysiere mindestens 5 Anzeigen, die seit über 6 Monaten aktiv sind. Liefere Einblicke in deren Inhalt, Format und mögliche Gründe für ihren anhaltenden Erfolg.',
            agent=self.meta_ads_analyst,
            expected_output="Eine detaillierte Analyse von mindestens 5 langfristig erfolgreichen Anzeigen aus der Meta Ads Library, einschließlich Inhalt, Format und Erfolgsfaktoren."
        )

        creative_recommendation_task = Task(
            description=f'Basierend auf der Trendanalyse, den Leistungseinblicken und den Erkenntnissen aus der Meta Ads Library, erstelle detaillierte Empfehlungen für Social-Media-Anzeigen-Kreationen für {analysis_area}. Gib jeweils 3 konkrete Vorschläge für Foto- und Video-Anzeigen. Für Fotoanzeigen: Beschreibe das Bild, den Anzeigentext und einen klaren Call-to-Action (CTA). Für Videoanzeigen: Beschreibe den Inhalt, die Länge, wichtige Szenen und den CTA. Stelle sicher, dass die Vorschläge direkt von einem Designer umgesetzt werden können.',
            agent=self.creative_director,
            expected_output="Detaillierte Empfehlungen für 3 Foto- und 3 Video-Anzeigen, einschließlich Beschreibungen von Bildern/Videos, Texten und CTAs, basierend auf den vorherigen Analysen."
        )

        analysis_crew = Crew(
            agents=[self.trend_analyst, self.ad_performance_analyst, self.meta_ads_analyst, self.creative_director],
            tasks=[trend_analysis_task, performance_analysis_task, meta_ads_analysis_task, creative_recommendation_task],
            process=Process.sequential,
            verbose=2
        )

        result = analysis_crew.kickoff()
        return result

if __name__ == "__main__":
    try:
        analyzer = SocialMediaAdAnalyzer()
        final_recommendation = analyzer.run_analysis()
        print(final_recommendation)

    except ValueError as e:
        print(f"Fehler: {e}")
        print("Bitte stellen Sie sicher, dass alle erforderlichen Umgebungsvariablen gesetzt sind.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

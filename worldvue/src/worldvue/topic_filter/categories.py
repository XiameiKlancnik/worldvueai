from typing import Dict, List

# Default topic categories (broad but actionable)
CATEGORIES: List[str] = [
    'government_policy',      # laws, regulation, public policy, administration
    'elections_politics',     # parties, campaigns, politicians
    'economy_markets',        # macroeconomy, inflation, unemployment, markets
    'business_companies',     # company news, earnings, M&A, industry trends
    'justice_courts',         # courts, rulings, investigations, rule of law
    'foreign_policy',         # diplomacy, sanctions, security, alliances
    'climate_energy',         # climate, energy policy, transition
    'health_public',          # public health, health policy
    'education',              # education policy, schools, universities
    'social_policy',          # civil rights, immigration, social programs
    'science_tech',           # science & tech policy, AI governance
    'environment',            # environment beyond climate policy
    'crime_incident',         # crime/accidents without policy angle
    'disasters_weather',      # weather, disasters, forecasts (non-policy)
    'sports',                 # sports events, results
    'entertainment_lifestyle' # entertainment, celebrity, lifestyle, culture reviews
]

DESCRIPTIONS: Dict[str, str] = {
    'government_policy': 'laws, regulation, public policy, executive/administrative decisions',
    'elections_politics': 'campaigns, politicians, parties, electoral processes',
    'economy_markets': 'macro indicators, markets, inflation, rates, central banks',
    'business_companies': 'company news, earnings, mergers, layoffs, industry trends',
    'justice_courts': 'courts, rulings, prosecutions, rule of law',
    'foreign_policy': 'diplomacy, sanctions, defense alliances, international security',
    'climate_energy': 'climate discourse, energy policy, transition',
    'health_public': 'public health policy and systems',
    'education': 'education policy, schools, universities',
    'social_policy': 'civil rights, immigration, social programs, protests with policy angle',
    'science_tech': 'science/technology policy, AI regulations',
    'environment': 'environmental issues beyond climate policy',
    'crime_incident': 'crime/accidents without a clear policy or public-interest angle',
    'disasters_weather': 'weather forecasts/alerts, disasters w/o policy context',
    'sports': 'sports fixtures, results, transfers',
    'entertainment_lifestyle': 'entertainment, celebrity, lifestyle, culture reviews'
}

__all__ = ['CATEGORIES', 'DESCRIPTIONS']

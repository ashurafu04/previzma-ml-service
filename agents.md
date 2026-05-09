# AGENTS.md — Previzma Backend

## Business Context

Previzma est une application web de prévision des ventes pour l'industrie B2B.
Le produit ne sert pas seulement à stocker des ventes : il transforme des données commerciales / ERP en informations décisionnelles exploitables.

Objectif métier :

- Aider les managers commerciaux à anticiper les ventes.
- Suivre les performances par segment client.
- Détecter les risques opérationnels.
- Simuler l'impact de décisions commerciales.

## Domain Model

Concepts métier principaux :

- `Company` : entreprise cliente, racine du modèle métier et du cloisonnement multi-tenant.
- `User` : utilisateur appartenant à une `Company`.
- `UserRole` : `ADMIN`, `MANAGER`, `ANALYST`.
- `Product` : produit industriel vendu.
- `ClientSegment` : segmentation B2B, par exemple `GRAND_COMPTE`, `PME`, `DISTRIBUTEUR`.
- `Sale` : vente historique rattachée à un produit et à un segment.
- `Forecast` : prévision générée par le service ML.
- `Simulation` : scénario What-If lancé par un utilisateur.
- `OperationalAlert` : alerte métier liée à un produit ou segment.

Vision fonctionnelle :

- `ADMIN` gère l'entreprise, les utilisateurs, les produits, les segments et l'import des données.
- `MANAGER` consulte le dashboard, les KPIs, les alertes, les prévisions et lance des simulations What-If.
- `ANALYST` interprète les résultats, compare les scénarios et analyse les tendances.

## Target Architecture

Architecture cible :

```text
Angular 18
  -> Spring Boot 3.x
      -> PostgreSQL/Supabase
      -> FastAPI/XGBoost ML service over HTTP
```

Ce repository contient uniquement le backend métier Spring Boot.

Le service FastAPI ML est un projet séparé. Il expose :

```text
POST /predict
POST /simulate
```

Spring Boot prépare les données métier, appelle FastAPI, puis sauvegarde les résultats en base.

## Repository Scope

Ce repo est responsable de :

- API métier REST.
- Authentification JWT.
- RBAC Spring Security.
- Cloisonnement applicatif par `companyId`.
- Persistance PostgreSQL/Supabase.
- Orchestration métier des forecasts et simulations.
- Client HTTP vers le service ML FastAPI.

Ce repo ne doit pas contenir :

- Modèle XGBoost.
- Code Python.
- Entrainement ML.
- Connexion directe du service ML à Supabase.
- Redis, Kafka, Docker ou autre infrastructure sans demande explicite.

## Backend Architecture Rules

Respecter package by feature.

Structure attendue par module :

```text
feature/
  Entity
  Controller
  Service
  Repository
  Request DTO
  Response DTO
```

Règles :

- Le backend Spring Boot doit rester orienté métier.
- Ne pas créer de CRUD générique qui ignore les relations du domaine.
- Respecter le flux `Controller -> Service -> Repository -> Entity`.
- Garder la logique métier dans les services, pas dans les controllers.
- Utiliser des DTOs dès qu'une entité contient des données sensibles ou des relations.
- Ne jamais retourner `password`, token ou secret dans une réponse JSON.
- Ne jamais écrire de secrets dans `application.properties`.
- Les secrets doivent venir de `.env` ou de variables d'environnement.

## Security Rules

Le backend utilise JWT et Spring Security.

Règles obligatoires :

- Les endpoints métier doivent être protégés par JWT sauf cas explicitement publics.
- Les règles de rôle doivent être déclarées avec `@PreAuthorize` ou équivalent clair.
- Les accès aux données doivent être limités à la `companyId` du JWT.
- Ne jamais utiliser un `companyId` libre du client sans vérification.
- Ne jamais exposer les données d'une autre `Company`.
- Pour une ressource liée à `Product`, `ClientSegment`, `Sale`, `Forecast`, `Simulation` ou `OperationalAlert`, vérifier le rattachement à la company courante.

Important :

```text
RBAC = rôle autorisé à faire une action.
Company scoping = données limitées à la company du JWT.
```

Les deux sont nécessaires.

## ML Integration Rules

Spring Boot ne fait pas de ML.

Le package Java lié au ML doit rester un client d'intégration HTTP, pas un moteur de prédiction.

Règles :

- FastAPI calcule les prévisions et simulations.
- Spring Boot prépare les données métier.
- Spring Boot appelle `/predict` pour générer un `Forecast`.
- Spring Boot appelle `/simulate` pour générer une `Simulation`.
- Spring Boot sauvegarde les résultats en base.
- Ne pas garder une transaction DB ouverte pendant un appel HTTP externe.

Contrat actuel :

```text
previzma.ml.base-url=${ML_BASE_URL:http://localhost:8000}
previzma.ml.predict-path=${ML_PREDICT_PATH:/predict}
previzma.ml.simulate-path=${ML_SIMULATE_PATH:/simulate}
```

## Error Handling And Validation

Règles :

- Utiliser `jakarta.validation` sur les request DTOs.
- Utiliser `@Valid` dans les controllers.
- Retourner des erreurs structurées via le global error handling.
- Utiliser des exceptions métier explicites plutôt que `RuntimeException` brute.
- Ne pas divulguer de détails sensibles dans les messages d'erreur.

## Testing And Verification

Pour chaque ticket :

- Lancer les tests Maven quand le code est modifié.
- Vérifier que le contexte Spring démarre.
- Ajouter des tests ciblés quand une nouvelle brique technique est introduite.
- Les tests fonctionnels Postman/UI peuvent être différés seulement si c'est explicitement décidé.
- Chaque module métier doit rester testable avec Postman et vérifiable dans Supabase.

Commande standard :

```powershell
.\mvnw.cmd test
```

Le projet cible Java 21.

## Development Contract

Pour chaque ticket PREV :

1. Lire le code existant avant de proposer ou modifier.
2. Respecter package by feature.
3. Garder les changements limités au ticket.
4. Préserver les relations métier.
5. Protéger les données sensibles.
6. Respecter JWT, RBAC et `companyId` scoping.
7. Ne pas ajouter d'infrastructure non demandée.
8. Privilégier une V1 simple, propre et défendable.
9. Lancer les tests utiles avant commit.
10. Ne pas commit sans validation explicite de l'utilisateur.

## Current Product Checkpoint

Etat architectural atteint :

```text
Supabase / schema métier V1       : en place
Spring Boot package by feature    : en place
Validation + error handling       : en place
JWT authentication                : en place
RBAC + company scoping            : en place
ML forecast client integration    : en place
ML simulation client integration  : en place
```

Les tests fonctionnels complets RBAC/RLS applicatif sont différés jusqu'a disponibilité de l'interface UI, sauf demande contraire.

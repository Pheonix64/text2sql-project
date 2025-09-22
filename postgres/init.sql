-- Activer l'extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Création de la table pour les indicateurs économiques de l'UEMOA
-- Les noms de colonnes ont été nettoyés pour être valides en SQL (pas d'espaces, accents, ou caractères spéciaux).
CREATE TABLE indicateurs_economiques_uemoa (
    date DATE NOT NULL,
    pib_nominal_milliards_fcfa REAL,
    poids_secteur_primaire_pct REAL,
    poids_secteur_secondaire_pct REAL,
    poids_secteur_tertiaire_pct REAL,
    taux_croissance_reel_pib_pct REAL,
    contribution_croissance_primaire REAL,
    contribution_croissance_secondaire REAL,
    contribution_croissance_tertiaire REAL,
    epargne_interieure_milliards_fcfa REAL,
    taux_epargne_interieure_pct REAL,
    taux_epargne_interieure_publique_pct REAL,
    investissement_milliards_fcfa REAL,
    taux_investissement_pct REAL,
    taux_investissement_public_pct REAL,
    taux_inflation_moyen_annuel_ipc_pct REAL,
    taux_inflation_glissement_annuel_pct REAL,
    recettes_totales_et_dons REAL,
    recettes_totales_hors_dons REAL,
    recettes_fiscales REAL,
    recettes_fiscales_pct_pib REAL,
    depenses_totales_et_prets_nets REAL,
    depenses_courantes REAL,
    investissements_sur_ressources_internes REAL,
    solde_primaire_base_sur_recettes_fiscales_pct REAL,
    solde_budgetaire_de_base REAL,
    solde_budgetaire_global_avec_dons REAL,
    solde_budgetaire_global_hors_dons REAL,
    encours_de_la_dette REAL,
    encours_de_la_dette_pct_pib REAL,
    service_de_la_dette_regle REAL,
    service_de_la_dette_interets REAL,
    exportations_biens_fob REAL,
    importations_biens_fob REAL,
    balance_des_biens REAL,
    compte_transactions_courantes REAL,
    balance_courante_sur_pib_pct REAL,
    balance_courante_hors_dons_publics REAL,
    balance_courante_hors_dons_sur_pib_pct REAL,
    solde_global_apres_ajustement REAL,
    financement_exceptionnel REAL,
    degre_ouverture_pct REAL,
    agregats_monnaie_actifs_exterieurs_nets REAL,
    agregats_monnaie_creances_interieures REAL,
    agregats_monnaie_creances_autres_secteurs REAL,
    agregats_monnaie_masse_monetaire_m2 REAL,
    actifs_exterieurs_nets_bceao_avoirs_officiels REAL,
    taux_couverture_emission_monetaire REAL
);

-- Ajout des commentaires pour enrichir le contexte du LLM
COMMENT ON TABLE indicateurs_economiques_uemoa IS 'Table contenant les principaux indicateurs macroéconomiques et financiers pour la zone UEMOA, sur une base annuelle.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.date IS 'Date de l''enregistrement, au format AAAA-MM-JJ. Représente le début de l''année de la donnée.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.pib_nominal_milliards_fcfa IS 'Produit Intérieur Brut nominal, en milliards de FCFA.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.poids_secteur_primaire_pct IS 'Poids du secteur primaire (agriculture, etc.) dans le PIB, en pourcentage (%).';
COMMENT ON COLUMN indicateurs_economiques_uemoa.poids_secteur_secondaire_pct IS 'Poids du secteur secondaire (industrie, etc.) dans le PIB, en pourcentage (%).';
COMMENT ON COLUMN indicateurs_economiques_uemoa.poids_secteur_tertiaire_pct IS 'Poids du secteur tertiaire (services, etc.) dans le PIB, en pourcentage (%).';
COMMENT ON COLUMN indicateurs_economiques_uemoa.taux_croissance_reel_pib_pct IS 'Taux de croissance annuel du PIB réel, en pourcentage (%).';
COMMENT ON COLUMN indicateurs_economiques_uemoa.taux_inflation_moyen_annuel_ipc_pct IS 'Taux d''inflation moyen annuel basé sur l''Indice des Prix à la Consommation (IPC), en pourcentage (%).';
COMMENT ON COLUMN indicateurs_economiques_uemoa.recettes_fiscales IS 'Total des recettes fiscales collectées, en milliards de FCFA.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.depenses_totales_et_prets_nets IS 'Total des dépenses de l''État, y compris les prêts nets, en milliards de FCFA.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.solde_budgetaire_global_avec_dons IS 'Solde budgétaire global, incluant les dons, en milliards de FCFA. Une valeur négative indique un déficit.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.encours_de_la_dette_pct_pib IS 'Encours total de la dette publique en pourcentage (%) du PIB nominal.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.exportations_biens_fob IS 'Valeur des exportations de biens (Franco à Bord), en milliards de FCFA.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.importations_biens_fob IS 'Valeur des importations de biens (Franco à Bord), en milliards de FCFA.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.balance_des_biens IS 'Solde commercial des biens (Exportations - Importations), en milliards de FCFA.';
COMMENT ON COLUMN indicateurs_economiques_uemoa.agregats_monnaie_masse_monetaire_m2 IS 'Masse monétaire M2, en milliards de FCFA.';

-- Transformation en Hypertable TimescaleDB pour optimiser les requêtes temporelles
SELECT create_hypertable('indicateurs_economiques_uemoa', 'date');

-- Chargement des données depuis le fichier CSV
-- NOTE: Le fichier est monté par Docker Compose sous le nom 'data.csv'
-- On spécifie le délimiteur, le format CSV avec en-tête, et comment traiter les valeurs vides.
COPY indicateurs_economiques_uemoa
FROM '/docker-entrypoint-initdb.d/data.csv'
DELIMITER ';'
CSV HEADER
NULL '';

-- Suppression de la dernière ligne si elle est vide (cas du fichier fourni)
DELETE FROM indicateurs_economiques_uemoa WHERE pib_nominal_milliards_fcfa IS NULL;

-- ====================================================================
-- SECTION SÉCURITÉ : PRINCIPE DU MOINDRE PRIVILÈGE
-- ====================================================================
CREATE USER llm_user WITH PASSWORD '/-+3Vd9$!D@12';
REVOKE ALL ON DATABASE economic_data FROM llm_user;
REVOKE ALL ON SCHEMA public FROM llm_user;
GRANT CONNECT ON DATABASE economic_data TO llm_user;
GRANT USAGE ON SCHEMA public TO llm_user;
-- Donner la permission de LECTURE SEULE sur la nouvelle table
GRANT SELECT ON indicateurs_economiques_uemoa TO llm_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO llm_user;
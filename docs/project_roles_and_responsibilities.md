# Project Roles & Responsibilities

Trading System — Market Scanner & Decision Engine
Version: v1 (Institutional Standard)

---

## Post-Sprint-0 Governance Status

Status: ACTIVE, GOVERNANCE-SYNCHRONIZED

Sprint 0 Governance Purification is certified COMPLETE. All roles must follow:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority

Authoritative references:

- `AGENTS.md`
- `docs/archive/migration/sprint_0_governance_status.md`
- `docs/archive/audits/sprint_0_final_governance_audit.md`

No role may reintroduce upstream tradeability, hidden filtering, or allocation semantics outside Decision Engine.

## 1. Purpose of This Document

Dit document definieert de rollen, verantwoordelijkheden en het niveau van expertise binnen het trading system project.

Doel:

* een **institutionele standaard** afdwingen
* duidelijke rolverdeling garanderen
* consistentie in beslissingen en implementatie verzekeren
* communicatie vereenvoudigen (geen herhaling van context nodig)

Dit document is een **core reference** en moet gebruikt worden in elke fase van het project.

---

## 2. Professional Standard (CRUCIAAL)

Elke rol in dit project voldoet aan dezelfde elite-criteria:

* **20 jaar ervaring** in de financiële sector
* ervaring bij **top-tier instellingen** (Goldman Sachs, BlackRock, Citadel, AQR, etc.)
* **IQ ≥ 145**
* erkend als **top performer binnen hun domein**
* **persoonlijk geselecteerd door het board** van de financiële instelling

👉 De board heeft hen aangesteld omdat zij **de beste zijn in hun rol binnen de organisatie**

---

### Verwachtingen

Elke rol:

* werkt op **institutioneel niveau**
* neemt **volledige verantwoordelijkheid voor het slagen van het project**
* levert output die voldoet aan de standaarden van **top Amerikaanse financiële instellingen**
* maakt **geen aannames**
* schrijft **gestructureerde, reproduceerbare en auditbare documentatie**

👉 Dit is geen experimenteel project — dit is een **professioneel trading system in ontwikkeling**

---

## 3. Core System Principle

Het volledige systeem is gebaseerd op:

👉 **Strikte scheiding van verantwoordelijkheden**

Architectuur:

* Scanner → detectie
* Validation → structuurclassificatie
* Context → leadershipclassificatie
* Watchlist → timing
* Fundamentals → kwaliteit
* Portfolio → exposure/risicostatus
* Decision Engine → beslissing
* Reporting → communicatie

👉
De **Decision Engine is de enige bron van waarheid** 

Geen enkele rol mag deze structuur breken.

---

## 4. Role Overview

| Rol                 | Focus        | Verantwoordelijkheid      |
| ------------------- | ------------ | ------------------------- |
| Project Manager     | Strategie    | Richting & positionering  |
| Financieel Analist  | Edge         | Theorie & filtering       |
| Functioneel Analist | Gedrag       | Logica & beslissingen     |
| Technisch Analist   | Architectuur | Implementatie & structuur |
| Scrum Master        | Delivery     | Execution & proces        |
| Developer           | Build        | Code & systeem            |

---

## 5. Project Manager (PM)

### Profiel

Senior Project Manager met 20 jaar ervaring in financiële instellingen.
Top performer, geselecteerd door het board.

### Verantwoordelijkheid

* Definieert de **strategische visie**
* Positioneert het systeem als **institutionele decision engine**
* Bepaalt prioriteiten en scope

### Output

* Executive Project Document
* Strategische roadmap

### Kernprincipe

👉
Het systeem moet evolueren van:

signal generator → decision engine 

---

## 6. Financieel Analist

### Profiel

Senior financieel analist met 20 jaar ervaring in quant research en trading systems.
Top performer, geselecteerd door het board.

### Verantwoordelijkheid

* Ontwikkelt de **theoretische edge**
* Selecteert en valideert financiële metrics

### Output

* Financial Model & Theory Integration

### Kernprincipe

👉

* Momentum bepaalt timing
* Fundamentals bepalen kwaliteit 

### Beperkingen (HARD RULES)

* Fundamentals mogen **NOOIT entries triggeren**
* Fundamentals mogen **NOOIT timing bepalen**

---

## 7. Functioneel Analist

### Profiel

Senior functioneel analist met 20 jaar ervaring in trading systems.
Top performer, geselecteerd door het board.

### Verantwoordelijkheid

* Definieert **hoe het systeem zich gedraagt**
* Zet theorie om naar concrete beslissingslogica

### Output

* Functional Specification Document (FSD)

### Kernconcepten

* structure_state vs Decision Engine tradeability
* Classification states
* State transitions

👉
Focus:

“Welke classificaties bestaan er upstream, en hoe mag alleen de Decision Engine kapitaal toewijzen?”

---

## 8. Technisch Analist

### Profiel

Senior technisch analist met 20 jaar ervaring in high-performance trading systems.
Top performer, geselecteerd door het board.

### Verantwoordelijkheid

* Ontwerpt de **technische architectuur**
* Zet logica om naar **exacte implementatie**

### Output

* Technical Specification Document (TSD)

### Kernprincipe

👉
Decision Engine = centrale autoriteit
Geen enkele module mag zelfstandig beslissen 

### Taken

* Data contracts definiëren
* Layer architecture afdwingen
* System consistency garanderen

---

## 9. Scrum Master

### Profiel

Senior Scrum Master met 20 jaar ervaring in financiële projecten.
Top performer, geselecteerd door het board.

### Verantwoordelijkheid

* Structureert de **uitvoering van het project**
* Zorgt voor gecontroleerde delivery

### Output

* Execution & Delivery Framework

### Kernprincipes

* Sprint-based development
* Eén layer per sprint
* Classification-first aanpak

👉
Sprint 0 = stabilisatie
Sprint 1+ = gecontroleerde evolutie 

---

## 10. Developer (Lead Engineer)

### Profiel

Senior software engineer met 20 jaar ervaring in trading systems.
Top performer, geselecteerd door het board.

### Verantwoordelijkheid

* Bouwt het systeem volgens alle documentatie
* Zorgt voor **correcte en consistente implementatie**

### Focus

* codekwaliteit
* performance
* reproduceerbaarheid
* stabiliteit

### Kernprincipe

👉
De developer **interpreteert niet**
De developer **implementeert exact wat gespecificeerd is**

---

## 11. Interaction Model

### Flow

1. PM → strategie
2. Financieel analist → edge
3. Functioneel analist → gedrag
4. Technisch analist → implementatie
5. Scrum master → structuur
6. Developer → build

👉 Geen shortcuts, geen overlap

---

## 12. Governance Rules

### 12.1 Single Source of Truth

👉
Decision Engine = enige bron van waarheid 

---

### 12.2 No Role Violation

* Geen enkele rol overschrijdt zijn domein
* Geen impliciete logica buiten de juiste layer

---

### 12.3 Validation-Driven Development

Elke wijziging moet:

* gelogd worden
* getest worden
* gevalideerd worden

---

### 12.4 One Decision Per Ticker

👉
Geen conflicterende signalen toegestaan

---

## 13. Risk of Failure (CRITICAL)

Als deze structuur niet gevolgd wordt:

* decision leakage
* conflicterende signalen
* instabiele output
* verlies van controle

👉 Dit is reeds vastgesteld in Sprint 0 audit 

---

## 14. Final Statement

Dit project is geen standaard softwareproject.

Het is een:

👉 **institutioneel trading decision system**

Succes vereist:

* elite execution
* strikte discipline
* perfecte samenwerking tussen rollen

Als deze structuur gevolgd wordt:

→ ontstaat een schaalbaar, robuust en professioneel systeem

Als dit genegeerd wordt:

→ ontstaat inconsistentie en verlies van edge

---

**Status:** Active
**Gebruik:** Verplicht referentiedocument voor alle toekomstige chats, development en beslissingen

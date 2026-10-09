"""
PawCare - Authoritative Veterinary Emergency Knowledge Base
Curated from reputable veterinary medicine institutions:
- American Veterinary Medical Association (AVMA)
- Cornell University College of Veterinary Medicine (Emergency & Critical Care)
- ASPCA Animal Poison Control Center (APCC)
- UC Davis School of Veterinary Medicine (Emergency Services)
- American Animal Hospital Association (AAHA)
- Merck Veterinary Manual (Emergency Medicine & Critical Care)
"""

EMERGENCY_KNOWLEDGE_BASE = [
    {
        "id": "breathing-difficulty",
        "title": "Breathing Difficulty, Respiratory Distress & Cyanosis",
        "category": "Respiratory",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Respiratory distress (dyspnea) occurs when a dog or cat cannot exchange adequate oxygen. In cats, open-mouth breathing is always a critical emergency. Blue, purple, or gray tongue/gums (cyanosis) indicates severe hypoxia.",
        "signs": [
            "Open-mouth breathing or gasping (especially critical in cats)",
            "Neck extended with elbows pointing outward ('orthopneic stance')",
            "Abdominal effort during breathing (stomach moves violently with chest)",
            "Loud wheezing, raspy rattling, or high-pitched squeaking",
            "Blue, purple, gray, or pale white gums or tongue",
            "Sudden collapse or weakness while attempting to breathe"
        ],
        "immediate_actions": [
            "Minimize all stress immediately; keep the environment completely quiet, dim, and calm (anxiety significantly spikes oxygen consumption).",
            "Loosen or remove all collars, harnesses, and bandages from the neck and chest immediately.",
            "Position the pet gently on their chest (sternal recumbency) with the neck extended straight forward to maximize airway diameter.",
            "Maintain cool, fresh air ventilation (turn on car air conditioning or open windows).",
            "Call the nearest 24/7 veterinary emergency hospital immediately so their clinical team can prepare an oxygen cage before your arrival."
        ],
        "things_to_avoid": [
            "DO NOT attempt mouth-to-snout rescue breathing unless the animal is completely in cardiopulmonary arrest and unresponsive.",
            "DO NOT compress, wrap, or place pressure on the chest or neck.",
            "DO NOT force the pet into an airtight container or carrier without ample ventilation.",
            "DO NOT force the pet to drink water or swallow oral medications; risk of fatal aspiration pneumonia.",
            "DO NOT delay transport to 'wait and see' if breathing improves."
        ],
        "vet_escalation": "Immediate emergency veterinary hospitalization is mandatory. Transport the pet in a comfortable position with direct vehicle air conditioning. Keep handling to the absolute minimum.",
        "clarifying_questions": [
            "Is the pet a dog or a cat, and are they breathing with their mouth open?",
            "What color are their gums and tongue (pink, pale, blue, or purple)?",
            "Did this start abruptly after eating, playing, or heat exposure?"
        ],
        "sources": [
            {
                "name": "American Veterinary Medical Association (AVMA)",
                "title": "Emergency Care & First Aid Guidelines for Pet Respiratory Distress",
                "organization": "AVMA Veterinary Emergency Resource Center"
            },
            {
                "name": "Cornell University College of Veterinary Medicine",
                "title": "Feline Respiratory Emergencies & Oxygen Therapy Protocols",
                "organization": "Cornell Companion Animal Hospital"
            }
        ]
    },
    {
        "id": "choking",
        "title": "Choking & Acute Airway Obstruction",
        "category": "Airway",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Foreign body lodging in the pharynx, larynx, or trachea preventing ventilation. Characterized by violent pawing at the mouth, gagging, inability to vocalize, and rapid progression to cyanosis or collapse.",
        "signs": [
            "Violent pawing at the mouth, frantic head shaking",
            "Silent gagging or retching without sound",
            "Inability to inhale, loud whistling stridor",
            "Salivating heavily with bulging eyes",
            "Blue, purple tongue transitioning to sudden limp collapse"
        ],
        "immediate_actions": [
            "Carefully open the mouth by pressing the lips over the upper teeth ONLY if the pet is calm enough not to bite reflexively.",
            "If a foreign object is clearly visible and loose in the front of the mouth, sweep it out sideways with a finger or blunt tweezers.",
            "For small dogs and cats: hold the pet upside down by the thighs for 5 seconds to allow gravity to assist dislodgement.",
            "For medium/large dogs: perform modified pet Heimlich maneuver — place hands just below the rib cage on the soft abdomen, deliver 3 to 5 rapid upward and forward thrusts.",
            "Begin immediate transit to the emergency clinic while continuing gentle airway checks."
        ],
        "things_to_avoid": [
            "DO NOT blindly poke fingers deep down the throat; this frequently pushes the foreign object into the trachea.",
            "DO NOT pull on protruding strings, threads, or needles caught in the throat (can lacerate esophagus or intestines).",
            "DO NOT waste more than 30-45 seconds attempting home removal if unsuccessful; immediately drive to the clinic."
        ],
        "vet_escalation": "Airway obstruction causes brain damage from hypoxia within minutes. Rush to the nearest emergency clinic immediately. Even if dislodged, throat swelling and secondary pulmonary edema require clinical evaluation.",
        "clarifying_questions": [
            "Can you see what object is stuck (ball, bone, toy, chew treat)?",
            "Is the pet still able to make sounds, cough, or move air?",
            "Is the pet conscious and responsive right now?"
        ],
        "sources": [
            {
                "name": "American Animal Hospital Association (AAHA)",
                "title": "Canine & Feline CPR and Airway Obstruction Clinical Protocols",
                "organization": "AAHA Emergency Critical Care Taskforce"
            },
            {
                "name": "UC Davis School of Veterinary Medicine",
                "title": "First Aid Protocols for Foreign Body Choking in Small Animals",
                "organization": "UC Davis Veterinary Medical Teaching Hospital"
            }
        ]
    },
    {
        "id": "severe-bleeding",
        "title": "Severe Bleeding, Arterial Puncture & Hemorrhage",
        "category": "Trauma",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Rapid arterial or venous blood loss resulting from deep lacerations, bites, vehicular trauma, or impalement. Uncontrolled bleeding leads rapidly to hypovolemic shock, organ failure, and cardiac arrest.",
        "signs": [
            "Pulsing or spurting blood (arterial) or continuous dark welling (venous)",
            "Blood rapidly soaking through towels or pooling on the floor",
            "Pale or white gums, rapid weak femoral pulse",
            "Cold extremities (paws, ears), hypothermia, progressive lethargy"
        ],
        "immediate_actions": [
            "Apply direct, continuous, firm pressure directly over the wound with a clean towel, gauze compress, or cloth.",
            "Maintain steady pressure for a full 5 minutes without lifting the cloth to check (lifting tears newly formed platelets).",
            "If blood soaks through, place additional cloths directly on top of the original layer; NEVER peel off the first layer.",
            "If bleeding is on a limb or paw, gently elevate the limb slightly above heart level while continuing direct pressure.",
            "Wrap the pet in a warm blanket to counteract hypovolemic shock and transport immediately."
        ],
        "things_to_avoid": [
            "DO NOT apply a tight improvised tourniquet unless explicitly instructed by a veterinarian (causes irreversible nerve death and gangrene).",
            "DO NOT remove deeply impaled objects (sticks, glass shards, metal) yourself; the object is actively plugging lacerated vessels.",
            "DO NOT pour hydrogen peroxide, alcohol, or powders into deep, actively bleeding wounds."
        ],
        "vet_escalation": "Severe hemorrhage leads to irreversible circulatory collapse. Keep firm pressure on the wound while transporting the pet directly to the nearest emergency hospital.",
        "clarifying_questions": [
            "Where is the bleeding located (limb, neck, chest, abdomen)?",
            "Is the blood spurting under pressure or steadily oozing?",
            "What color are your pet's gums (healthy pink, pale white, or gray)?"
        ],
        "sources": [
            {
                "name": "Merck Veterinary Manual",
                "title": "Management of Hemorrhage and Hypovolemic Shock in Dogs and Cats",
                "organization": "Merck Veterinary Clinical Medicine"
            },
            {
                "name": "AVMA",
                "title": "First Aid for Bleeding & Wounds in Companion Animals",
                "organization": "AVMA Emergency Care Center"
            }
        ]
    },
    {
        "id": "xylitol-poisoning",
        "title": "Poisoning: Xylitol (Birch Sweetener) Ingestion in Dogs",
        "category": "Toxicology",
        "species": "dog",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Xylitol (wood sugar / birch bark extract) is an artificial sweetener found in sugar-free gum, peanut butter, baked goods, and chewable vitamins. In dogs, it stimulates an enormous insulin surge causing profound life-threatening hypoglycemia within 15-60 minutes, followed by acute hepatic necrosis (liver failure).",
        "signs": [
            "Vomiting within 15 to 30 minutes of ingestion",
            "Profound weakness, stumbling, 'drunken' wobbly gait (ataxia)",
            "Lethargy, muscle tremors, seizures, collapse",
            "Pale gums, hypothermia, coma"
        ],
        "immediate_actions": [
            "Take the packaging, ingredient label, or gum wrapper immediately to show the vet the exact xylitol concentration.",
            "Check the time of ingestion and estimate how many pieces/grams were consumed.",
            "If the dog is conscious and able to swallow safely, rubbing a small amount of maple syrup, honey, or sugar water onto their gums can temporarily slow blood sugar crash while en route.",
            "Contact your emergency vet or ASPCA Animal Poison Control (888-426-4435) immediately while in transit."
        ],
        "things_to_avoid": [
            "DO NOT attempt to induce vomiting at home without veterinary instructions, especially if the pet is wobbly or lethargic (high aspiration risk).",
            "DO NOT wait for seizures or collapse before seeking care; liver damage begins rapidly.",
            "DO NOT assume small amounts are safe; as little as 0.1g/kg causes hypoglycemia, and 0.5g/kg causes fatal acute liver failure."
        ],
        "vet_escalation": "Xylitol toxicity is an immediate medical emergency requiring intravenous dextrose infusion and intensive hepatoprotective therapy. Proceed to an emergency clinic immediately.",
        "clarifying_questions": [
            "How long ago did your dog consume the substance?",
            "What brand of gum, candy, or peanut butter was it?",
            "Is your dog showing stumbling, tremors, or weakness right now?"
        ],
        "sources": [
            {
                "name": "ASPCA Animal Poison Control Center (APCC)",
                "title": "Xylitol Toxicity in Canine Patients: Hypoglycemia & Hepatic Failure",
                "organization": "ASPCA APCC Toxicological Reference"
            },
            {
                "name": "Cornell University College of Veterinary Medicine",
                "title": "Sweeteners and Canine Toxicity: Emergency Management Protocol",
                "organization": "Cornell Animal Health Diagnostic Center"
            }
        ]
    },
    {
        "id": "lily-poisoning-cats",
        "title": "Poisoning: Lily Plant Ingestion & Toxicity in Cats",
        "category": "Toxicology",
        "species": "cat",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "True lilies (Lilium species like Easter, Stargazer, Tiger lilies) and Daylilies (Hemerocallis) are extremely toxic to cats. Every part—petals, leaves, pollen, stem, and even water from the vase—causes fatal acute tubular necrosis and acute renal failure.",
        "signs": [
            "Vomiting, drooling, and loss of appetite within 1 to 3 hours",
            "Lethargy, depression, hiding",
            "Increased urination followed by complete cessation of urine (anuria)",
            "Dehydration, seizures, collapse within 24 to 72 hours"
        ],
        "immediate_actions": [
            "Remove the cat from the plant immediately.",
            "If pollen is on the cat's fur or face, gently wash it off with lukewarm water and a damp cloth to prevent the cat from grooming and swallowing it.",
            "Bring the plant, flower, or a photo of the plant to the emergency clinic for botanic identification.",
            "Transport to an emergency veterinary clinic immediately."
        ],
        "things_to_avoid": [
            "DO NOT wait to see if the cat becomes sick; once kidney values spike at 24-48 hours, renal failure is often irreversible.",
            "DO NOT attempt home induction of vomiting in cats (causes severe esophageal irritation and aspiration).",
            "DO NOT allow the cat to lick paws or fur that contacted lily pollen."
        ],
        "vet_escalation": "Immediate aggressive intravenous fluid diuresis within the first 6-12 hours of exposure is the only proven method to prevent fatal kidney failure in cats.",
        "clarifying_questions": [
            "What type of lily was it (Easter, Tiger, Daylily, Peace lily)?",
            "Did the cat chew a leaf, petal, or get pollen on their fur?",
            "How many hours ago was the cat near the bouquet or plant?"
        ],
        "sources": [
            {
                "name": "ASPCA Animal Poison Control Center (APCC)",
                "title": "Feline Lily Nephrotoxicity Guidelines",
                "organization": "ASPCA Animal Poison Control Center"
            },
            {
                "name": "UC Davis School of Veterinary Medicine",
                "title": "Acute Renal Failure Caused by Lily Ingestion in Feline Patients",
                "organization": "UC Davis Veterinary Teaching Hospital"
            }
        ]
    },
    {
        "id": "chocolate-poisoning",
        "title": "Poisoning: Chocolate & Caffeine (Theobromine) Ingestion",
        "category": "Toxicology",
        "species": "all",
        "triage_level": "URGENT",
        "urgency_badge": "🟠 URGENT — Contact Veterinarian Promptly",
        "urgency_code": "urgent",
        "summary": "Chocolate contains methylxanthines (theobromine and caffeine). Dogs metabolize theobromine extremely slowly. Toxicity severity depends on chocolate type: Baker's chocolate and dark chocolate are highly toxic; milk chocolate is moderate; white chocolate has low theobromine but causes pancreatitis.",
        "signs": [
            "Vomiting, diarrhea, and marked restlessness / pacing",
            "Elevated heart rate (tachycardia), panting, increased thirst",
            "Muscle rigidity, twitching, tremors",
            "Cardiac arrhythmias, hyperthermia, seizures in severe cases"
        ],
        "immediate_actions": [
            "Find the chocolate packaging to check exact cocoa percentage and total weight ingested (ounces or grams).",
            "Record your pet's approximate body weight and the time of ingestion.",
            "Contact your veterinarian or emergency clinic with these figures for an exact toxicity calculation.",
            "Keep the pet in a cool, quiet room away from loud stimuli that trigger arrhythmias."
        ],
        "things_to_avoid": [
            "DO NOT administer salt to induce vomiting (causes fatal hypernatremia / brain swelling).",
            "DO NOT administer hydrogen peroxide without explicit veterinary instruction and precise dosing.",
            "DO NOT assume small dark chocolate bars are harmless; a 20lb dog can suffer cardiac toxicity from just 1.5 oz of baker's chocolate."
        ],
        "vet_escalation": "If significant dark/baker's chocolate was ingested, or if your pet displays tremors, pacing, or racing heart, proceed immediately to veterinary care for gastric decontamination and activated charcoal.",
        "clarifying_questions": [
            "What type of chocolate was eaten (milk, dark, baker's, cocoa powder)?",
            "How much was consumed (ounces, grams, or number of squares)?",
            "How much does your dog weigh, and how long ago was it eaten?"
        ],
        "sources": [
            {
                "name": "ASPCA Animal Poison Control Center (APCC)",
                "title": "Methylxanthine Toxicosis in Dogs: Chocolate & Caffeine",
                "organization": "ASPCA Veterinary Toxicology Division"
            },
            {
                "name": "AVMA",
                "title": "Pet Emergency Guidance: Chocolate Ingestion in Companion Animals",
                "organization": "AVMA Public Health & Emergency Resources"
            }
        ]
    },
    {
        "id": "rodenticide-poisoning",
        "title": "Poisoning: Rodenticide & Rat/Mouse Bait Ingestion",
        "category": "Toxicology",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Commercial rodenticides come in three lethal classes: Anticoagulants (depletes Vitamin K1 causing internal hemorrhage 3-7 days later), Bromethalin (causes neurotoxic cerebral edema, paralysis), and Cholecalciferol (Vitamin D3 causing acute kidney failure).",
        "signs": [
            "Anticoagulant type: Lethargy, pale gums, nosebleeds, bloody urine, coughing blood, bruised skin (delayed 2-5 days)",
            "Bromethalin type: Incoordination, muscle tremors, hyperexcitability, paralysis, seizures (within 2-24 hours)",
            "Cholecalciferol type: Increased thirst/urination, vomiting, dehydration, acute renal failure"
        ],
        "immediate_actions": [
            "Collect the bait box, brand name, active ingredient EPA registration number, and remaining bait block.",
            "Check the time of ingestion and pet weight.",
            "Call the emergency veterinary hospital immediately to notify them of the active ingredient.",
            "Proceed directly to the clinic for timely gastric decontamination."
        ],
        "things_to_avoid": [
            "DO NOT wait for clinical bleeding to start; by the time bruising or coughing blood appears, internal blood clotting factors are completely exhausted.",
            "DO NOT purchase over-the-counter Vitamin K supplements from health food stores (Vitamin K3 is ineffective and toxic; only prescription veterinary Vitamin K1 works).",
            "DO NOT guess the active ingredient; the three classes require completely different clinical treatments."
        ],
        "vet_escalation": "Immediate emergency care is required. Antidote therapy (prescription Vitamin K1) or decontamination must begin before cellular toxins absorb.",
        "clarifying_questions": [
            "Do you have the packaging or brand name of the rat/mouse poison?",
            "What color was the bait block or pellet (green, blue, pink)?",
            "Did the pet eat the bait directly or eat a poisoned rodent?"
        ],
        "sources": [
            {
                "name": "ASPCA Animal Poison Control Center (APCC)",
                "title": "Rodenticide Toxicity Classifications & Antidote Protocols",
                "organization": "ASPCA Toxicological Data Archive"
            },
            {
                "name": "Merck Veterinary Manual",
                "title": "Anticoagulant and Non-Anticoagulant Rodenticide Poisoning",
                "organization": "Merck Veterinary Clinical Reference"
            }
        ]
    },
    {
        "id": "heatstroke",
        "title": "Heat-Related Emergency & Heatstroke (Hyperthermia)",
        "category": "Environmental",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Elevated core body temperature (>40°C / 104°F) from confinement in warm vehicles, intense exercise, or high humidity. Brachycephalic breeds (Pugs, Bulldogs, Persians) are at extreme risk. Causes multiorgan failure, DIC (coagulation breakdown), and cerebral edema.",
        "signs": [
            "Excessive, frantic panting with thick ropey saliva",
            "Bright brick-red, dark purple, or dry tacky gums",
            "Stumbling, disorientation, glazed expression, collapse",
            "Vomiting, bloody diarrhea, rectal temperature > 40°C (104°F)"
        ],
        "immediate_actions": [
            "Move the pet immediately into shade, air conditioning, or near a strong cool fan.",
            "Apply cool (tap water, NOT ice) wet towels to the paws, groin, underbelly, and neck.",
            "Pour cool tap water over the body and direct a fan continuously over wet areas to drive evaporative cooling.",
            "Offer small sips of cool water if the pet is conscious and eager to drink voluntarily.",
            "Stop cooling once rectal temperature reaches 39.4°C (103°F) to prevent hypothermic rebound shock."
        ],
        "things_to_avoid": [
            "DO NOT use ice, ice cubes, or freezing ice baths (causes rapid peripheral vasoconstriction that traps heat inside vital organs).",
            "DO NOT submerge the head or force water into an unconscious pet's mouth.",
            "DO NOT wrap the pet in damp towels and leave them on without changing (they quickly become hot insulators).",
            "DO NOT skip the vet visit if the pet seems better; delayed internal hemorrhage and kidney damage manifest 24-48 hours later."
        ],
        "vet_escalation": "Heatstroke is a life-threatening systemic crisis. Continue gentle evaporative cooling while transporting the pet directly to the emergency hospital with vehicle AC running.",
        "clarifying_questions": [
            "Was the pet in a parked car, outdoors in hot sun, or exercising heavily?",
            "What breed is your pet (is it a flat-faced breed like a Bulldog or Pug)?",
            "Is the pet conscious, walking, or collapsed?"
        ],
        "sources": [
            {
                "name": "American Animal Hospital Association (AAHA)",
                "title": "Hyperthermia & Heatstroke Emergency Resuscitation in Small Animals",
                "organization": "AAHA Critical Care Protocols"
            },
            {
                "name": "AVMA",
                "title": "Warm Weather Pet Safety & Heatstroke Management",
                "organization": "AVMA Veterinary Emergency Center"
            }
        ]
    },
    {
        "id": "seizures",
        "title": "Seizures, Convulsions & Status Epilepticus",
        "category": "Neurological",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Sudden abnormal electrical discharges in the brain causing grand mal convulsions, paddling of legs, jaw snapping, hypersalivation, and loss of consciousness. Seizures lasting longer than 3 minutes, or 2+ seizures in 24 hours (cluster seizures), cause irreversible brain hyperthermia and death.",
        "signs": [
            "Violent stiffening, rhythmic leg paddling, jaw chomping",
            "Frothing at the mouth, involuntary urination or defecation",
            "Unresponsiveness during the episode (ictus)",
            "Post-seizure disorientation, pacing, temporary blindness (post-ictal phase)"
        ],
        "immediate_actions": [
            "Clear away all hard, sharp, or heavy furniture to prevent secondary head/limb trauma.",
            "Place soft pillows, blankets, or cushions on the floor surrounding the pet.",
            "Turn off bright overhead lights, turn off televisions, and eliminate noise.",
            "Time the seizure with a stopwatch; record a brief 10-second phone video if safe to show the neurologist.",
            "Speak in a calm, quiet, soothing voice as the pet regains awareness."
        ],
        "things_to_avoid": [
            "DO NOT place hands, fingers, or objects near the pet's mouth (pets DO NOT swallow their tongues, but bite with tremendous reflex force).",
            "DO NOT attempt to hold down or forcefully restrain the convulsing body.",
            "DO NOT offer food, treats, or water until the pet is completely alert and steady (choking hazard)."
        ],
        "vet_escalation": "CRITICAL EMERGENCY if: the seizure lasts > 3 minutes, or the pet has 2+ seizures in 24 hours, or fails to regain consciousness between episodes. Rush to an emergency facility immediately.",
        "clarifying_questions": [
            "How many minutes did the seizure last?",
            "Has your pet had another seizure today or in the past?",
            "Is your pet known to be on seizure medication or exposed to toxins?"
        ],
        "sources": [
            {
                "name": "Cornell University College of Veterinary Medicine",
                "title": "Canine & Feline Epilepsy and Status Epilepticus Emergency Protocol",
                "organization": "Cornell Companion Animal Neurology"
            },
            {
                "name": "Merck Veterinary Manual",
                "title": "Diagnosis and Emergency Management of Convulsions and Seizures",
                "organization": "Merck Veterinary Reference"
            }
        ]
    },
    {
        "id": "collapse-unconsciousness",
        "title": "Sudden Collapse, Syncope & Loss of Consciousness",
        "category": "Cardiovascular",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Sudden loss of postural tone and responsiveness. May result from cardiac arrhythmias, severe internal hemorrhage (hemoabdomen, ruptured splenic tumor), Addisonian crisis, hypoglycemia, or severe anaphylaxis.",
        "signs": [
            "Sudden falling to the side, limp muscles, inability to stand",
            "Unresponsiveness to vocal calls or physical pinching of toes",
            "Extremely pale, white, or blue gums",
            "Very weak or absent pulse in the femoral artery (inner thigh)"
        ],
        "immediate_actions": [
            "Check for breathing by observing the chest wall or feeling for air on your cheek near nostrils.",
            "Check for a femoral pulse on the inside of the upper rear thigh.",
            "Gently extend the neck straight forward and ensure the mouth is free of vomit or tongue obstruction.",
            "Keep the pet's body horizontal or with hind legs elevated slightly on a towel to promote blood flow to the brain.",
            "Cover with a warm blanket and transport immediately."
        ],
        "things_to_avoid": [
            "DO NOT vigorously shake, slap, or aggressively jostle the unconscious pet.",
            "DO NOT pour liquids or place oral remedies into the mouth of an unresponsive animal.",
            "DO NOT perform chest compressions if the pet has a detectable heartbeat."
        ],
        "vet_escalation": "Sudden collapse is a grave medical emergency. Immediate emergency veterinary evaluation is essential to stabilize cardiovascular and neurological function.",
        "clarifying_questions": [
            "Is the pet breathing and can you feel a heartbeat in their chest?",
            "Did the collapse happen during exercise, excitement, or rest?",
            "What color are their gums right now?"
        ],
        "sources": [
            {
                "name": "UC Davis School of Veterinary Medicine",
                "title": "Triage and Emergency Resuscitation of the Collapsed Patient",
                "organization": "UC Davis Veterinary Medical Teaching Hospital"
            },
            {
                "name": "AVMA",
                "title": "Emergency First Aid: Unconsciousness & Shock",
                "organization": "AVMA Emergency Care Protocols"
            }
        ]
    },
    {
        "id": "bloat-gdv",
        "title": "Gastric Dilatation-Volvulus (GDV / Bloat) in Dogs",
        "category": "Gastrointestinal",
        "species": "dog",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Critical Surgical Emergency",
        "urgency_code": "emergency",
        "summary": "The stomach fills with gas (dilatation) and twists along its axis (volvulus), cutting off blood supply to the stomach and spleen and blocking venous return to the heart. Common in deep-chested breeds (Great Danes, German Shepherds, Boxers, Standard Poodles, Labs). Fatal within hours without emergency decompression and surgery.",
        "signs": [
            "Non-productive retching (trying desperately to vomit but producing only white foamy slime)",
            "Distended, visibly swollen, or drum-tight, painful abdomen",
            "Severe restlessness, pacing, inability to sit or lie down comfortably",
            "Drooling heavily, rapid shallow panting",
            "Pale gums, weakness, and rapid progression to shock"
        ],
        "immediate_actions": [
            "Recognize non-productive retching + distended abdomen as an immediate surgical crisis.",
            "Do NOT wait or attempt any home remedy.",
            "Call the emergency clinic immediately and say: 'I suspect my dog has GDV / Bloat, I am on the way.'",
            "Transport the dog immediately on a flat, padded surface."
        ],
        "things_to_avoid": [
            "DO NOT give gas drops (Simethicone), water, food, or milk (does not untwist the stomach and delays surgical prep).",
            "DO NOT attempt to decompress the stomach yourself with needles or tubes.",
            "DO NOT wait until morning; mortality rate increases by 10-15% with every hour of delay."
        ],
        "vet_escalation": "This is one of the most time-critical surgical emergencies in veterinary medicine. Proceed immediately to a 24-hour veterinary surgical facility.",
        "clarifying_questions": [
            "Is your dog retching without bringing up any stomach contents?",
            "Is the abdomen visibly enlarged, tight, or painful to the touch?",
            "What breed is your dog (is it a deep-chested breed)?"
        ],
        "sources": [
            {
                "name": "American Animal Hospital Association (AAHA)",
                "title": "Canine Gastric Dilatation-Volvulus: Recognition and Emergency Management",
                "organization": "AAHA Emergency Guidelines"
            },
            {
                "name": "Cornell University College of Veterinary Medicine",
                "title": "Emergency Surgical Protocols for Canine GDV",
                "organization": "Cornell Companion Animal Hospital"
            }
        ]
    },
    {
        "id": "feline-urinary-blockage",
        "title": "Feline Urethral Obstruction (Blocked Cat)",
        "category": "Urological",
        "species": "cat",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Urethral blockage by urinary crystals, mucous plugs, or spasm, primarily affecting male cats. The cat cannot empty the bladder. Toxic waste and potassium accumulate, leading to hyperkalemia, cardiac arrest, bladder rupture, and death within 24-48 hours.",
        "signs": [
            "Repeatedly visiting the litter box, straining with little or zero urine produced",
            "Crying, howling, or growling in the litter box",
            "Excessive, frantic licking of the genital area",
            "Vomiting, profound lethargy, hiding, hard painful golf-ball/baseball-sized bladder",
            "Cold paws, slow heart rate, hypothermia"
        ],
        "immediate_actions": [
            "Check the litter box immediately for actual wet urine clumps.",
            "Keep the cat calm and in a secure, padded carrier.",
            "Contact your emergency vet clinic immediately.",
            "Transport immediately for emergency urinary catheterization."
        ],
        "things_to_avoid": [
            "DO NOT squeeze, press, or palpate the cat's lower belly (can rupture a tense, fragile bladder into the abdominal cavity).",
            "DO NOT mistake this for constipation; male cats straining in the box are almost always urinary blocked.",
            "DO NOT give over-the-counter urinary supplements, cranberry, or human pain meds.",
            "DO NOT wait overnight to see if the cat passes urine."
        ],
        "vet_escalation": "A blocked cat is an immediate medical emergency. Potassium toxicity causes fatal heart arrhythmias. Rush to an emergency clinic right away.",
        "clarifying_questions": [
            "Is your cat a male or female?",
            "Is any urine coming out at all (even small drops), or zero?",
            "How many hours has the cat been straining or crying in the box?"
        ],
        "sources": [
            {
                "name": "Cornell University College of Veterinary Medicine",
                "title": "Feline Urethral Obstruction: Diagnosis & Emergency Decompression",
                "organization": "Cornell Feline Health Center"
            },
            {
                "name": "AVMA",
                "title": "Feline Lower Urinary Tract Disease & Emergency Blockage",
                "organization": "AVMA Veterinary Clinical Guidelines"
            }
        ]
    },
    {
        "id": "trauma-fractures",
        "title": "Physical Trauma, Falls from Heights & Suspected Fractures",
        "category": "Trauma",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Blunt force trauma from car impact, high-rise balcony falls ('high-rise syndrome'), or severe dog fights. Frequently accompanied by internal hemorrhage, pneumothorax, diaphragmatic rupture, or spinal injuries.",
        "signs": [
            "Inability to bear weight, abnormal limb angles, bone protruding through skin",
            "Whimpering, crying out when touched, rapid shallow breathing",
            "Bleeding from mouth, nose, or rectum",
            "Dilated pupils, inability to move hind legs (paresis/paralysis)"
        ],
        "immediate_actions": [
            "Approach with extreme caution; pets in intense pain bite reflexively regardless of bond.",
            "Keep the pet's spine, neck, and body as straight and level as possible.",
            "Slide a firm flat board, baking tray, or thick folded blanket underneath the pet to use as a stretcher.",
            "Cover open wounds and exposed bone gently with a clean, dry cloth; do not press.",
            "Transport gently and avoid sharp turns or sudden braking."
        ],
        "things_to_avoid": [
            "DO NOT attempt to set splints or push exposed bone back inside.",
            "DO NOT give human pain medications (Tylenol/Paracetamol causes fatal Heinz body anemia in cats; Advil/Ibuprofen causes fatal gastric ulceration and renal necrosis in dogs).",
            "DO NOT allow the pet to walk, stand, or jump."
        ],
        "vet_escalation": "Trauma cases often have life-threatening internal injuries that are invisible on the outside. Proceed immediately to an emergency clinic for radiography and stabilization.",
        "clarifying_questions": [
            "Did the pet fall from a height or get hit by a vehicle?",
            "Are there open wounds or bone visible through the skin?",
            "Is the pet able to move all four limbs?"
        ],
        "sources": [
            {
                "name": "UC Davis School of Veterinary Medicine",
                "title": "Emergency Stabilization of Small Animal Trauma and Fractures",
                "organization": "UC Davis Veterinary Emergency & Critical Care"
            },
            {
                "name": "AAHA",
                "title": "Companion Animal Spinal Trauma & Fracture First-Response Guidelines",
                "organization": "AAHA Emergency Guidelines"
            }
        ]
    },
    {
        "id": "eye-emergencies",
        "title": "Eye Trauma, Corneal Injury & Proptosis",
        "category": "Ophthalmology",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Acute ocular injuries including proptosis (eyeball displaced out of eye socket), deep corneal scratches/ulcers, penetrating foreign bodies, or chemical splashes. Permanent blindness can occur without urgent treatment within hours.",
        "signs": [
            "Eyeball protruding from socket (proptosis, common in Pugs, Shih Tzus)",
            "Squinting, blepharospasm (inability to open the eye), pawing at the eye",
            "Cloudy, milky, or blue haze over the cornea, red blood in the anterior chamber",
            "Excessive tearing, yellow/green discharge, or visible scratch"
        ],
        "immediate_actions": [
            "If the eyeball is protruding: gently cover the eye with a clean gauze or towel soaked in sterile saline (or clean lukewarm water) to keep the cornea moist. NEVER let it dry out.",
            "Put an Elizabethan collar (cone) on the pet immediately to stop pawing.",
            "If a harmless chemical splashed in the eye, flush gently with sterile saline eyewash for 5 minutes.",
            "Transport immediately to the veterinary clinic."
        ],
        "things_to_avoid": [
            "DO NOT attempt to push an eyeball back into the socket yourself.",
            "DO NOT apply human eye drops, steroid drops, or Visine (steroids cause rapid corneal perforation if an ulcer is present).",
            "DO NOT attempt to pull out objects impaled in the eyeball."
        ],
        "vet_escalation": "Ocular emergencies require immediate veterinary care within 1-2 hours to preserve vision and save the eye globe.",
        "clarifying_questions": [
            "Is the eyeball bulging out of the socket or inside the eye?",
            "Did a chemical, cat scratch, or object cause the injury?",
            "Is the pet actively rubbing or pawing at the eye?"
        ],
        "sources": [
            {
                "name": "Merck Veterinary Manual",
                "title": "Ocular Emergencies in Dogs and Cats: Clinical Protocols",
                "organization": "Merck Veterinary Ophthalmology"
            },
            {
                "name": "AVMA",
                "title": "Emergency Management of Companion Animal Eye Injuries",
                "organization": "AVMA Veterinary Guidelines"
            }
        ]
    },
    {
        "id": "anaphylaxis-insect-bites",
        "title": "Anaphylaxis, Severe Allergic Reactions & Insect Stings",
        "category": "Immunology",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Acute hypersensitivity reaction triggered by bee/wasp stings, spider bites, vaccines, or medications. In dogs and cats, anaphylaxis can cause profound hypotension, gastrointestinal shock, and respiratory collapse.",
        "signs": [
            "Sudden severe facial swelling (muzzle, lips, eyelids, ears)",
            "Hives (urticaria) erupting across the skin, intense itching",
            "Difficulty breathing, loud wheezing, or stridor",
            "Sudden severe vomiting, diarrhea, weakness, or collapse"
        ],
        "immediate_actions": [
            "If a stinger is visible on the skin, gently scrape it off with a credit card edge (do not squeeze with tweezers, which injects more venom).",
            "Apply a cold pack wrapped in a cloth to the swollen area for 10 minutes to soothe pain and swelling.",
            "Keep the pet calm and quiet.",
            "Contact your veterinarian immediately for weight-appropriate antihistamine dosing or proceed directly to emergency if breathing is compromised."
        ],
        "things_to_avoid": [
            "DO NOT administer human allergy medications containing decongestants (Pseudoephedrine or Phenylephrine are fatal to pets).",
            "DO NOT delay if swelling is near the throat or if the pet is vomiting/weak.",
            "DO NOT squeeze insect stingers."
        ],
        "vet_escalation": "If your pet shows difficulty breathing, collapse, vomiting, or rapidly spreading swelling, this is systemic anaphylactic shock requiring immediate epinephrine, corticosteroids, and IV fluids.",
        "clarifying_questions": [
            "Did you see a bee, wasp, or insect bite the pet?",
            "Is the swelling confined to the muzzle or is the pet having trouble breathing?",
            "Has the pet vomited or collapsed?"
        ],
        "sources": [
            {
                "name": "American Animal Hospital Association (AAHA)",
                "title": "Emergency Diagnosis and Treatment of Anaphylaxis in Small Animals",
                "organization": "AAHA Critical Care Committee"
            },
            {
                "name": "Cornell University College of Veterinary Medicine",
                "title": "Insect Envenomation and Acute Allergic Reactions in Dogs and Cats",
                "organization": "Cornell Companion Animal Health"
            }
        ]
    },
    {
        "id": "severe-vomiting-diarrhea",
        "title": "Severe Vomiting, Hematemesis & Acute Dehydration",
        "category": "Gastrointestinal",
        "species": "all",
        "triage_level": "URGENT",
        "urgency_badge": "🟠 URGENT — Contact Veterinarian Promptly",
        "urgency_code": "urgent",
        "summary": "Frequent, repeated vomiting or bloody diarrhea causing rapid fluid and electrolyte depletion. Can indicate bowel obstruction, parvovirus, pancreatitis, or toxic ingestion.",
        "signs": [
            "Vomiting more than 3 times in a few hours, unable to keep water down",
            "Blood in vomit (bright red or dark coffee-ground appearance)",
            "Bloody diarrhea (melena or frank raspberry-jam stool)",
            "Dry, sticky gums, sunken eyes, skin tenting (delayed skin elasticity >2s)",
            "Extreme lethargy or depression"
        ],
        "immediate_actions": [
            "Withhold solid food for 4-6 hours to rest the stomach lining.",
            "Offer small controlled sips of water (a few teaspoons every 30 minutes) rather than allowing large gulps.",
            "Check gum moisture: touch the gums; if tacky or sticky, dehydration has set in.",
            "Note the frequency, volume, color, and presence of foreign materials.",
            "Contact your veterinarian for prompt evaluation."
        ],
        "things_to_avoid": [
            "DO NOT give human anti-diarrheal medications (Pepto-Bismol contains bismuth subsalicylate, toxic to cats and dogs; Imodium/Loperamide causes neurotoxicity in MDR1 breeds).",
            "DO NOT force food, milk, or broth on a nauseated animal.",
            "DO NOT wait if your pet is a puppy or kitten; young animals develop fatal hypoglycemia and dehydration within hours."
        ],
        "vet_escalation": "If accompanied by blood, severe lethargy, abdominal pain, or in puppies/kittens, immediate veterinary care is required.",
        "clarifying_questions": [
            "How many times has your pet vomited today?",
            "Is there blood in the vomit or diarrhea?",
            "Could your pet have swallowed a toy, bone, cloth, or trash?"
        ],
        "sources": [
            {
                "name": "Merck Veterinary Manual",
                "title": "Acute Vomiting and Diarrhea in Small Animals",
                "organization": "Merck Veterinary Gastroenterology"
            },
            {
                "name": "AVMA",
                "title": "Gastrointestinal Emergencies & Dehydration Monitoring",
                "organization": "AVMA Pet Health Guide"
            }
        ]
    },
    {
        "id": "foreign-body-ingestion",
        "title": "Foreign Object Ingestion (Bones, Strings, Toys, Batteries)",
        "category": "Gastrointestinal",
        "species": "all",
        "triage_level": "URGENT",
        "urgency_badge": "🟠 URGENT — Contact Veterinarian Promptly",
        "urgency_code": "urgent",
        "summary": "Ingestion of non-food objects. Linear foreign bodies (yarn, string, ribbon) in cats saw through intestinal walls; batteries cause corrosive liquefaction necrosis; coins (pennies) cause zinc toxicity; sharp bones cause perforation.",
        "signs": [
            "Witnessed ingestion of an object",
            "Persistent retching, vomiting after eating or drinking",
            "Abdominal pain, praying posture (front paws down, rear up)",
            "String visible under the tongue or protruding from rectum"
        ],
        "immediate_actions": [
            "Note the exact object swallowed, size, material, and time of ingestion.",
            "If battery or magnet: proceed to emergency clinic IMMEDIATELY (battery acid causes tissue burn within 2 hours).",
            "Call your vet clinic immediately: if caught within 1-2 hours, endoscopy or safe emesis can remove it before surgical laparotomy is required.",
            "Keep the pet calm and withhold further food."
        ],
        "things_to_avoid": [
            "DO NOT pull on string or ribbon protruding from the mouth or anus (can slice through intestines like a cheese wire).",
            "DO NOT induce vomiting if the object was sharp, battery, caustic, or large without explicit vet direction.",
            "DO NOT wait days hoping large objects will pass."
        ],
        "vet_escalation": "Bowel obstruction leads to intestinal necrosis and sepsis. Contact your veterinarian promptly for imaging (radiographs/ultrasound).",
        "clarifying_questions": [
            "What specific object was swallowed, and how long ago?",
            "Is the pet vomiting or able to keep water down?",
            "Are there any batteries, magnets, or sharp materials involved?"
        ],
        "sources": [
            {
                "name": "American Animal Hospital Association (AAHA)",
                "title": "Linear & Non-Linear Foreign Body Obstruction in Dogs and Cats",
                "organization": "AAHA Surgical Guidelines"
            },
            {
                "name": "UC Davis School of Veterinary Medicine",
                "title": "Endoscopic vs. Surgical Management of Foreign Bodies",
                "organization": "UC Davis Small Animal Surgery"
            }
        ]
    },
    {
        "id": "burns-electric-shock",
        "title": "Burns, Scalds & Electrical Cord Shock",
        "category": "Trauma",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Thermal burns from boiling water/grease, chemical burns, or electric shock from chewing cords (common in puppies/kittens). Electrical burns frequently trigger delayed life-threatening non-cardiogenic pulmonary edema (fluid in lungs) 1-12 hours later.",
        "signs": [
            "Charred skin, singed fur, blistering, weeping open dermis",
            "For electrical shock: burned lips/tongue, sudden collapse, respiratory distress, coughing up pink froth"
        ],
        "immediate_actions": [
            "For electric shock: UNPLUG the electrical cord before touching the pet.",
            "For thermal burns: apply cool tap water compresses (NOT ice) for 10-15 minutes to halt thermal burn progression.",
            "Cover burned areas loosely with a clean, dry cloth or sterile plastic wrap.",
            "Keep the pet warm and transport immediately."
        ],
        "things_to_avoid": [
            "DO NOT apply ice, butter, grease, ointments, or toothpaste to burns (traps heat and breeds infection).",
            "DO NOT pop blisters or peel off adhered charred material.",
            "DO NOT assume a pet that chewed an electric cord is safe just because they survived the initial jolt; lung edema develops hours later."
        ],
        "vet_escalation": "Burns risk severe infection and fluid loss. Electrical shock requires urgent veterinary monitoring for pulmonary edema and cardiac arrhythmias.",
        "clarifying_questions": [
            "Was the injury caused by hot liquid, flame, chemical, or an electrical cord?",
            "Is the pet coughing or having trouble breathing?",
            "Are there burns visible on the tongue, gums, or skin?"
        ],
        "sources": [
            {
                "name": "Merck Veterinary Manual",
                "title": "Thermal Burns and Electrical Injury in Companion Animals",
                "organization": "Merck Veterinary Emergency Medicine"
            },
            {
                "name": "AVMA",
                "title": "Emergency First Aid for Burns and Electrical Accidents",
                "organization": "AVMA Emergency Care Guide"
            }
        ]
    },
    {
        "id": "snake-bites",
        "title": "Snake Bites & Venomous Envenomation",
        "category": "Toxicology",
        "species": "all",
        "triage_level": "EMERGENCY",
        "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
        "urgency_code": "emergency",
        "summary": "Envenomation from pit vipers (rattlesnakes, copperheads, water moccasins) or elapids (coral snakes). Pit viper venom causes rapid local tissue necrosis, coagulopathy (systemic bleeding), severe shock, and organ failure.",
        "signs": [
            "Sudden severe swelling, intense pain, two puncture marks leaking bloody discharge",
            "Bruising, black/purple discoloration around the bite site",
            "Rapid panting, weakness, vomiting, muscle fasciculations, collapse"
        ],
        "immediate_actions": [
            "Keep the pet as still and calm as humanly possible; increased heart rate spreads venom rapidly through the bloodstream.",
            "Carry the pet to the car; do NOT let them walk.",
            "Keep the bite site at or slightly below heart level.",
            "Remove collars or harnesses immediately (facial bites swell massively within 15 minutes and choke airways).",
            "Call the nearest emergency hospital immediately to confirm they have antivenom (antivenin) in stock."
        ],
        "things_to_avoid": [
            "DO NOT cut the wound, suck out venom, or apply commercial venom extractors.",
            "DO NOT apply ice or cold packs (causes concentrated tissue necrosis and loss of limb).",
            "DO NOT apply a tourniquet.",
            "DO NOT attempt to catch or kill the snake (photograph safely from distance if possible)."
        ],
        "vet_escalation": "Venomous snake bites are a critical emergency requiring antivenin, IV fluid resuscitation, and intensive pain control.",
        "clarifying_questions": [
            "Where was the pet bitten (face, paw, leg)?",
            "Did you see the snake (what markings or colors)?",
            "How rapidly is the swelling spreading?"
        ],
        "sources": [
            {
                "name": "UC Davis School of Veterinary Medicine",
                "title": "Pit Viper Envenomation and Antivenin Protocols in Small Animals",
                "organization": "UC Davis Emergency Care Center"
            },
            {
                "name": "ASPCA Animal Poison Control Center (APCC)",
                "title": "Snakebite Envenomation Management Guidelines",
                "organization": "ASPCA Poison Control Resource"
            }
        ]
    },
    {
        "id": "hypothermia-cold",
        "title": "Hypothermia & Severe Cold Exposure",
        "category": "Environmental",
        "species": "all",
        "triage_level": "URGENT",
        "urgency_badge": "🟠 URGENT — Contact Veterinarian Promptly",
        "urgency_code": "urgent",
        "summary": "Subnormal body temperature (<37°C / 98.6°F) resulting from prolonged exposure to cold weather, immersion in cold water, or systemic shock. Causes bradycardia, respiratory depression, and cardiac arrest.",
        "signs": [
            "Violent shivering followed by cessation of shivering (severe stage)",
            "Pale, gray, or blue gums; cold, stiff paws and ears",
            "Extreme lethargy, slow shallow breathing, weak pulse"
        ],
        "immediate_actions": [
            "Move the pet indoors into a warm room.",
            "If wet, dry thoroughly with warm, dry towels.",
            "Wrap in several warm dry blankets, including paws and head.",
            "Place warm water bottles wrapped in towels against the pet's abdomen and chest.",
            "Check temperature and transport to veterinary care."
        ],
        "things_to_avoid": [
            "DO NOT use direct heating pads or hair dryers on bare skin (causes severe thermal burns).",
            "DO NOT warm extremities before warming the core (causes peripheral vasodilation and shock).",
            "DO NOT rub frostbitten ears or paws (causes tissue damage)."
        ],
        "vet_escalation": "Severe hypothermia requires active internal rewarming, warmed IV fluids, and cardiac monitoring.",
        "clarifying_questions": [
            "How long was the pet exposed to cold weather or water?",
            "Is the pet still shivering or have they stopped shivering?",
            "What is their current rectal temperature if measured?"
        ],
        "sources": [
            {
                "name": "AVMA",
                "title": "Cold Weather Pet Safety & Hypothermia Management",
                "organization": "AVMA Companion Animal Guidelines"
            },
            {
                "name": "Merck Veterinary Manual",
                "title": "Hypothermia and Frostbite in Small Animals",
                "organization": "Merck Veterinary Critical Care"
            }
        ]
    }
]

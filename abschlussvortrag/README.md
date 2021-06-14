# Titelfolie
Vielen Dank, Prof. Turau, für die nette Einleitung.
Dann würde ich einfach mal sagen: "Los gehts!".

# Motivation und Ziele
Vor 4 Wochen hatte Mian in seinen Abschlussvortrag über die diskrete Indoor-Lokalisierung mit Hilfe von FFNNs auf Basis von Sensoren geredet. 
Bei der Lokalisation wird die Position von einem Gerät oder Nutzer in einem Koordinatensystem bestimmt.
Ähnlich wie der Orientierungssinn von Menschen und Tieren, sollen keine konkreten Koordinaten gefunden werden, 
sondern diskrete Standorte unterschieden werden.

Mian hatte bereits FFNNs mit simulativ erzeugten Daten auf Basis von drei Sensoren untersucht: Accelerometer, Gyroskop und Lichtsensor.
In dieser Arbeit wurden weitere Sensoren für die diskrete Standortbestimmung benutzt, aus denen ausgewählte Features extrahiert wurden. 
Zusätzlich, zu den FFNNs, wurden Entscheidungsbäume untersucht, da diese potenziell effizienter sind als FFNNs.

# Ansatz zur diskreten Standortbestimmung I
Zunächst werden simulativ erfasste Daten als Standorte beschriftet.
Schematisch stellt die linke Zeichnung eine Route da, in der die hellblauen Bereiche beispielsweise Förderbänder und die weißen Kreise diskrete Standorte.
Auf dieser Route bewegt sich die Sensorenbox.
Selbstverständlich sind die evaluierten Szenarien in dieser Arbeit etwas komplexer.
Dabei kann eine Route als zyklischen Graph dargestellt werden, aus dem insgesamt drei Möglichkeiten hervorgehen, um Standorte zu kodieren.
1. Es werden nur die Knoten kodiert und alles andere wird als unbekannter Standort beschriftet.
2. Es werden nur die Kanten kodiert, das wäre äquivalent zu dem Ansatz den Mian verwendet hatte.
3. Oder, es werden sowohl Knoten und Kanten als Standorte kodiert.
In dieser Arbeit wird der erste und dritte Ansatz untersucht.

# Ansatz zur diskreten Standortbestimmung II
Die Standorte sind dann durch ihre Sensordaten unterscheidbar, indem jeder Standort eine einzigartige Menge an Features besitzt.
Features sind Attribute und Eigenschaften der Sensordaten. 
Dies ist in diesem Bild illustriert.
Standort 1 hat eine besonders hohe Lautstärke.
Standort 2 hat keines der Features.
Standort 3 ist in Reichweite eines WLAN-Zugangspunktes.
Standort 4 hat magnetische Interferenzen.

Diese Sensordaten werden dann von einer Sensorbox aufgenommen, die sich auf dieser Route bewegt.
Anhand der aufgenommen Sensordaten werden Features extrahiert, die von einem ML-Modell genutzt werden, um den momentanen Standort zu bestimmen.

# Sensoren
Rechts in dem Bild ist eine Route zu erkennen, mit der Sensordaten simulativ in CoppeliaSim erfasst wurden.
Gezeigt wird ein Förderband-Szenario auf dem sich die Sensorenbox bewegt.
Bei den Standortmarkierungen werden die aufgenommen Daten automatisch mit der Kante beschriftet, auf der sich die Sensorenbox befindet.
CoppeliaSim bietet nur Accelerometer-, Gyroskop und Lichtintensitätsdaten an, ohne dass komplexe Erweiterungen verwendet werden.
Aus diesem Grund wurden die rot markierten Sensordaten mit einem Modell in einem anschließenden Weiterverarbeitungsschritt ergänzt.

# Modell
Das in dieser Arbeit verwendete Modell soll Standorte erkennen, UND unterscheiden können, ob es sich in einer Anomalie befindet.
Als eine Anomalie werden Standorte bezeichnet, die nicht in der von dem Standortbestimmungsmodell gelernt wurden. 
Zunächst werden aus den aufgenommen Sensordaten Features extrahiert, die von einem ML-Modell zur Standortbestimmung benutzt werden, um den momentanen Standort zu bestimmen. 
Mit ML-Modell wird entweder ein Entscheidungsbaum basierter Klassifizierer gemeint, oder ein FFNN.
Die gestrichelte Rückwärtskante signalisiert, dass es zwei ML-Modell Varianten gibt.
Eines, welches den bestimmte Standort als Feature benutzt und eines welches dieses Feature nicht nutzt, 
d. h. wir unterscheiden insgesamt vier Modelle: Jeweils Entscheidungsbaum basierte Klassifizierer und FFNNs, mit UND ohne Ruckwärtskante.
Anschließend wird mit dem bestimmten Standort und historischen Standortdaten von einem weiteren ML-Modell eingeschätzt, 
ob es sich in einer Anomalie befindet.

# Feature-Extrahierung
Insgesamt werden 38 Features aus den Sensordaten extrahiert, die in fünf Gruppen eingeordnet werden können.
Diese Gruppen von Features werden auf ein Datenfenster von 3 Einträgen angewendet, d. h. die letzten drei Vektoren von Sensordaten werden verwendet.
Die erste Gruppe ist die Standardabweichung, welche eine relative Änderung im Datenfenster ausdrückt.
Die zweite und dritte Gruppe drückt Extrema im Datenfenster aus.
Die vierte Gruppe ist der Durchschnitt und mit der 5. Gruppe ist die letzte Eingabe gemeint.

# Training der ML-Modelle
Das Training der ML-Modelle ist abhängig davon, ob das ML-Modell eine Rückwärtskante nutzt oder nicht.
Wenn es eine Rückwärtskante nutzt, ist es nötig, dass der propagierte Fehler von dem ML-Modell gelernt wird, 
d. h. dass das ML-Modell lernen muss, mit dem von sich selbst verursachten Fehler umzugehen.
Dafür werden die Trainingsdaten in 20 Partitionen unterteilt, die in einen der drei Trainingsphasen verwendet werden.
Die ersten fünf Partitionen werden in der Aufwärmphase verwendet. 
Hier werden noch keine Daten verwendet, die von dem ML-Modell selbst bestimmt wurden.
In den folgenden 10 Partitionen wird ein zunehmend wachsender Anteil der nächsten Partition von dem ML-Modell selbst beschriftet.
Diese wird dann zu der Menge hinzugefügt, mit der die nächste Iteration des ML-Modells trainiert wird.
Dies wird wiederholt, bis mit der 15. Partition trainiert wurde.
Nun können wir die übrigen 5 Partitionen als Validationsdaten nutzen, auf Basis dessen wir die wichtigkeit der verwendeten Features bewerten können.
Unwichtige Features können dann, in einer Auswahlphase, aus der Feature-Menge entfernen, wodurch ein schlankeres Modell trainiert werden kann.
Dieser Schritt ist notwendig, da die Wichtigkeit der Features von dem Einsatzszenario abhängig ist.
Wenn das ML-Modell KEINE Rückwärtskante nutzt, ist dieser Prozess nicht notwendig, da der propagierte Fehler nicht gelernt werden muss.

# Klassifizierungsgenauigkeit über Standortkomplexitäten
Es wurden 256 verschiedene ML-Modelle auf bis zu 4 Routen trainiert mit jeweils dem Standortkodierungsverfahren, das nur Knoten kodiert und das Verfahren, das Knoten und Kanten kodiert.
Auf dem Graphen werden die besten ML-Modelle zu jeder Standortkomplexität verglichen.
Die Standortkomplexität drückt die Anzahl der Standorte aus.
Eingezeichnet sind Entscheidungswälder und FFNNs mit und ohne Rückwärtskante sowie Mians Modelle.

Zunächst ist zu sehen, dass die Entscheidungswälder deutlich bessere Klassifizierungsergebnisse über alle Standortkomplexitäten im Vergleich zu den FFNNs erzielen.
Dabei sind die ML-Modelle ohne Rückwärtskante besser als die ML-Modelle mit Rückwärtskante.
Die Entscheidungswälder skalieren sehr gut mit steigender Standortkomplexität, mit jedem zusätzlichen Standort verringert sich die Klassifizierungsgenauigkeit im Schnitt um 0,1 Prozentpunkte,
wohingegen die FFNNs 0,24 Prozentpunkte verlieren.
Mit dem Standortkodierungsverfahren, dass nur Knoten kodiert, wurden bessere Klassifizierungsergebnisse erzielt, als mit dem Verfahren, dass Knoten und Kanten kodiert.

Mians WFBNN, nutzt eine Rückwärtskante und das WFFNN nutzt keine Rückwärtskante.
Das "W" steht für "windowed" und deutet auf ein sensorisches Gedächtnis hin.
Mian hat Daten von 6, 9 und 14 Standorten aufgenommen. 
Es hat sich bei ihm ebenfalls herausgestellt, dass bessere Klassifizierungsergebnisse ohne Rückwärtskante erzielt werden können.
Verglichen werden die Ergebnisse von Mians Windowed-FFNN mit 14 Standorten und die ML-Modelle, dieser Arbeit, mit 17 Standorten.
Mians WFFNN erzielte eine Klassifizierungsgenauigkeit von 94,51%. 
Der beste Entscheidungswald dieser Arbeit erzielte 96,45%. Das sind 1,94 Prozentpunkte mehr.
Das beste FFNN erzielte 93,46%. Das sind 1,05 Prozentpunkte weniger.
Anzumerken ist, dass die FFNNs dieser Arbeit deutlich kleiner sind als die WFFNNs Mians arbeit. Dazu später mehr.

Es hat sich herausgestellt, dass eine maximale Baumhöhe von 16 Vergleichen, sowie kleine Waldgrößen ausreichend sind.
Die FFNNs erzielen keine besseren Ergebnisse mit mehr Schichten, aber mit mehr Neuronen pro verdeckte Schicht können die Klassifizierungsgenauigkeiten verbessert werden.

# Wichtigkeit von Features I
Um die Wichtigkeit von Features einzuschätzen, können zwei Methoden eingesetzt werden.
Die erste Methode ist nur auf Entscheidungsbäume anwendbar, wobei der Effekt der gewählten Entscheidungsregeln auf die Trainingsdaten analysiert wird.
Die zweite Methode kann auf alle ML-Modelle angewendet werden.
Dabei wird die Testmenge modifiziert, indem iterativ einzelne Features verändert werden.
Anschließend wird der resultierende Fehler im Vergleich zur originalen Testmenge verglichen.
Und je größer der Fehler, desto wichtiger ist das Feature.
Modifizierungsmethoden sind z. B. die Permutation oder Nullung von Features.

Hier wird die Permutationswichtigkeit illustriert.
Aus der Testmenge werden für jedes Feature eine Testmenge generiert, hier illustrieren die geometrischen Formen jeweils ein Feature.
Dabei wird jeweils nur das Feature Spaltenweise permutiert, wie in den rot unterstrichenden Spalten zu sehen ist.
Dann werden die Klassifizierungsgenauigkeiten dieser Testmengen zur originalen Testmenge verglichen.

# Permutationswichtigkeit - ML-Modelle mit Rückwärtskante
Hier wird die Permutationswichtigkeit der ML-Modelle MIT Rückwärtskante verglichen.
Insgesamt ist der Entscheidungswald (hier grün) robuster gegenüber Permutation als das FFNN (hier blau).
Sie gewichten aber zum größten Teil die gleichen Features.
Auffällig ist, dass die Features der Rückwärtskante KEINEN Effekt haben.
Das liegt daran, dass die ML-Modelle in der Evaluation diese Features selbst setzen, wodurch eine Permutation keinen Einfluss hat.
Würde dies nicht geschehen, verursachen die Features einen Fehler von bis zu 80%.
Verwendet werden hauptsächlich die Standardabweichung, die Minima und Maxima.
Insbesondere Licht, Ausrichtung zum Magnetfeld (Heading) und Lautstärke (volume) wird verwendet.
Zudem haben die WLAN-Zugangspunkte einen großen Einfluss und besonders das FFNN gewichtet die WLAN-Zugangspunkte sehr stark.

# Permutationswichtigkeit - ML-Modelle ohne Rückwärtskante
Hier wird die Permutationswichtigkeit der ML-Modelle OHNE Rückwärtskante verglichen.
Die Gewichtung des Entscheidungswaldes ist weitestgehend gleich zu dem Entscheidungswald mit Rückwärtskante, nur haben die einzelnen Features einen größeren Fehler verursacht.
Insbesondere das Minimum der Lichtintensität und die Standardabweichung zur Ausrichtung des Magnetfelds sind wichtig.
Für das FFNN hat sich die Wichtigkeit der Extrema und Standardabweichung für die Lichtintensität erhöht.
Zudem werden immer noch WLAN-Zugangspunkte und Standardabweichung zur Ausrichtung zum Magnetfeld genutzt.
Interessant ist die große Wichtigkeit für die Extrema der Ausrichtung zum Magnetfeld, da diese Features eigentlich keinen Sinn ergeben, 
da die Ausrichtung abhängig ist von der Ausrichtung der Sensorenbox, die sich im Laufe der Route ändern kann.

# Fehlertoleranz
Bei der Fehlertoleranz wurden die Fehler gemessen, wenn in den Testmengen Sensorwerte genullt wurden.
Dies wirkt sich damit auf die Features aus, die in diesem Fall dadurch ebenfalls genullt sind.
Zudem wurde auch getestet, ob ein Rauschen eine Auswirkung hat, diese Werte wurden aber ausgelassen, da kein Fehler gemessen wurde bei einem 5% Rauschen.
Auf der linken Seite sind die Fehler zu den genullten Testmengen von den ML-Modellen mit Rückwärtskante und auf der rechten Seite von den ML-Modellen ohne Rückwärtskante.

Auffällig im Vergleich zur Permutationswichtigkeit ist der Fehler, wenn der Temperatursensor genullt wurde.
In der Permutationswichtigkeit hatte dies kaum Einfluss auf den Entscheidungswald, hier ist der Fehler aber 15,15%-Punkte.
Im simulierten Szenario gibt es nur wenige Wärme- und Kältequellen, d. h. zum größten Teil wird die Umgebungstemperatur gemessen.
An einigen wenigen Stellen wird eine andere Temperatur gemessen, d. h. eine Permutation hat kaum Effekt, weil die Werte zum größten Teil ähnlich sind.
Es gibt nur eine Kältequelle im Szenario, die die Umgebungstemperatur senkt, weswegen ich vermute, dass der Fehler aus diesem Grund so groß ist, 
da die Nullung stark auf diesen Standort hinweist.

Unerwartet war außerdem, dass der Fehler des FFNNs mit Rückwärtskante bei der Ausrichtung zum Magnetfeld so groß ist, im Vergleich zu allen anderen ML-Modellen.
Die Permutationswichtigkeit hätte dies nicht vermuten lassen, im gegenteil hätte der Fehler des anderen FFNNs größer sein müssen.

Der Fehler durch die WLAN-Zugangspunkte hingegen entspricht den Erwartungen der Permutationswichtigkeit bei FFNNs.
Insbesondere, wenn man bedenkt, dass die WLAN-Zugangspunkte 14,7% aller Features ausmachen.

Der durchschnittliche Fehler war bei den Entscheidungswäldern am geringsten.
Es ist aber zu erwarten, dass bei einer vergleichbaren Klassifizierungsgenauigkeit auch ein vergleichbarer Fehler entsteht, wie bei dem Entscheidungswald mit Rückwärtskante und dem FFNN ohne RÜckwärtskante.

# Ressourcennutzung
In der Natur der Entscheidungswälder liegt, dass sie sehr effizient sind aber dafür viel mehr Programmspeicher benötigen als FFNNs.
Die Programmgröße skaliert bei beiden ML-Modellen mit der Standortkomplexität.
Im Vergleich zu Mian sind die FFNNs dieser Arbeit zwischen 54% und 97,6% kleiner und dafür nur marginal schlechter, also ca. 70KB bis 3KB.
Entscheidungswälder hingegen benötigen bei gleicher Standortkomplexität bis zu 720% MEHR Programmspeicher, also ca. 1MB.
Es können aber kleinere Entscheidungswälder gefunden werden, die sogar kleiner sein können, dafür aber nicht genau so gute Klassifizierungsgenauigkeiten erzielen.

# Anomalieerkennung
Bei der Anomalieerkennung wird versucht zu erkennen, ob sich die Sensorenbox an einem nicht trainierten Standort befindet.
Da es unendlich viele Möglichkeiten von Anomalien gibt, wurde eine Sensoren basierte Lösung ausgeschlossen.
Stattdessen wird versucht das Verhalten des ML-Modells zur Standortbestimmung einzuschätzen.

# Summierte Klassifizierungsgenauigkeit in Datenfenster
In dieser Abbildung wird die summierte Wahrscheinlichkeit in einem Datenfenster von 25 Einträgen angegeben, 
von dem Standort, dass vom dem ML-Modell mit der höchsten Wahrscheinlichkeit eingeschätzt wird.
Zu sehen sind drei sehr deutliche Extrema, in denen eine hohe Unsicherheit des ML-Modells ausgedrückt wird.
In diesem einfachen Szenario könnte vermutlich eine gleitender Mittelwert benutzt werden, um das Problem adequat zu lösen.
Allerdings haben komplexere Routen viele von diesen Pseudeo Anomalien (zeige auf nicht so deutliche Peaks), weswegen eine komplexere Lösung notwendig ist.

# Anzahl Standortänderungen in Datenfenster
Wenn die Sensorenbox sich in einer Anomalie befindet, wird davon ausgegangen, dass sie oft den momentan Standort ändert. 
Dies ist in dieser Grafik zu sehen, da bei den Anomalien (zeige auf Peaks) ungewöhnlich viele Standortänderungen, in einem Datenfenster von 25 Einträgen, zu beobachten sind.

# Permutationswichtigkeit
Aus diesen Indikatoren sind vier Features motiviert:
1. Abweichung zu den durchschnittlichen Standortänderungen
2. Abweichung zu der durchschnittlichen Standortwahrscheinlichkeit
3. Ob eine Topologieverletzung vorliegt, d. h. ob die Sensorenbox dem gelernten Pfad nicht folgt.
4. Standardabweichung der Top 5 wahrscheinlichsten Standorte des ML-Modells

Die wichtigsten Features sind diese, die die Sicherheit für die Klassifizierungsergebnisse der ML-Modelle ausnutzen.
Die anderen Features scheinen eher unwichtig zu sein.

# Klassifizierungsergebnisse
FFNNs waren nicht in der Lage trainiert zu werden.
Sie haben lediglich gelernt immer keine Anomalie auszugeben, was aber auch den Großteil der Daten ausmacht.
Entscheidungswälder konnten bis zu 52,52% der Anomalien erkennen, wobei 2,95% fälschlich als Anomalien erkannt wurden.
Die Genauigkeit scheint abhängig von der Standortbestimmungsrate und der Standortkomplexität zu sein.
Je mehr Standorte und je bessere Klassifizierungsgenauigkeiten in dieser Standortkomplexität, desto besser ist das resultierende ML-Modell der Anomalieerkennung.

Hier sieht man einen Ausschnitt der Klassifizierungsergebnisse eines Entscheidungswaldes.
Grüne Punkte zeigen die tatsächliche Anomalie, blaue Punkte die Einschätzung des ML-Modells.
Wie zu sehen ist, sind die meisten Erkennungen einer Anomalie bei einer tatsächlichen Anomalie.
Mit einem Schwellenwert könnten die Falsch-Positiven Ergebnisse möglicherweise eliminiert werden.

# Erkenntnisse dieser Arbeit
Entscheidungswälder skalieren besser als FFNNs mit steigender Standortkomplexität.
Dabei sind die ML-Modelle ohne Rückwärtskante besser als die mit Rückwärtskante.
Es konnten 98,62% bei 9 Standorten klassifiziert werden und 87,35% bei 102 Standorten.
Im Schnitt verringert sich die Klassifizierungsgenauigkeit um 0,1 Prozentpunkte pro zusätzlichen Standort.
Es konnten nur 52,58% der Anomalien erkannt werden, dafür aber mit einer geringen Fehlerrate.
Ein kleines Datenfenster ist ausreichend und ein sensorisches Gedächtnis ist nicht notwendig.
Aus diesem Grund sind die Modelle sehr klein und können auf Mikrocontrollern implementiert werden.
Entscheidungswälder benötigen deutlich weniger Operationen als FFNNs.
Es hat sich gezeigt, dass mit der Standortkodierungsmethode, die nur Knoten kodiert, bessere Klassifizierungsergebnisse erzielt werden konnten,
als mit der Methode, die Knoten und Kanten kodiert.
Die Wichtigkeit der Features ist abhängig vom Einsatzszenario.
Als sinnvolle Methoden, um die Wichtigkeit einzuschätzen haben sich die Permutationswichtigkeit oder Nullung ergeben, 
aber auch andere Modifizierungen wie z. B. ein Rauschen wäre denkbar.

## OLD
# Wichtigkeit von Features II
In diesem Bild wird die erste Methode illustriert.
Die weiße Box unter den Entscheidungsregeln zeigt die Beschriftungen der in diesem Knoten enthaltenden Trainingsdaten.
Durch jede Entscheidungsregel wird die Reinheit (engl. Purity) erhöht.
Die Wichtigkeit wird durch den Zuwachs der Reinheit bestimmt, den jedes Feature in einer Entscheidungsregel bringt.
In dieser Arbeit wurde dieser Ansatz aber nicht verwendet, da er nicht auf FFNNs angewendet werden konnte.

# Motivation und Ziele
Dabei wird zwischen Indoor- und Outdoor-Lokalisation unterschieden.
Ein bekanntes Outdoor Verfahren ist z. B. GPS, welches in Indoor Szenarien aber oft nicht aufgrund der Signalstärke Einsetzbar ist.
Aus diesem Grund wird bei der Indoor-Lokalisation Infrastruktur aufgebaut, um eine Lokalisation zu ermöglichen.
Die Infrastruktur ist nötig, um die Position des Geräts oder Nutzers eindeutig zu bestimmen.
Oft ist aber eine so gute Auflösung gar nicht nötig.

Anstatt die Position des Geräts zu bestimmen, kann das Gerät auch einfach seine eigene Position bestimmen.
Ein Beispiel aus der Natur ist der Orientierungssinn von Menschen und Tieren, die anhand von Orientierungspunkten feststellen an welchem Standort sie sich befinden.
Dies ist eine Form der diskreten Standortbestimmung, in der keine konkrete Position bestimmt wird, sondern nur der Ort anhand von Orientierungspunkten.

Für ML-Modelle können diese Orientierungspunkte durch aufgenommene Sensorwerte dargestellt werden auf Basis dessen diskrete Standorte unterschieden werden.
Mian hatte bereits FFNNs mit simulativ erzeugten Daten auf Basis von drei Sensoren untersucht: Accelerometer, Gyroskop und Lichtsensor.
Diese Arbeit untersucht zusätzlich Entscheidungsbäume, da sie potenziell effizienter sind als FFNNs.
Außerdem wurden noch mehr Sensoren verwendet, aus denen ausgewählte Features extrahiert wurden.
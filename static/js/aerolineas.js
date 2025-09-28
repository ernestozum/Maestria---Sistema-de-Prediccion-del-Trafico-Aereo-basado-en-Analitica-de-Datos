function renderAerolineasChart(){

    const loading = document.getElementById("loadingAerolineas");
    const noData = document.getElementById("noDataAerolineas");
    const canvas = document.getElementById("vuelosPorAerolineaChart");
    const topNSelect = document.getElementById("topNSelect");

    if (!canvas) return; // Evita errores si no está en la vista actual

    let chart; // Para poder destruirlo si ya existe

    function fetchAndRender(topN = 10) {
        loading.classList.remove("hidden");
        noData.classList.add("hidden");

        fetch("/api/estadisticas-aerolineas")
            .then((response) => {
                if (!response.ok) throw new Error("Error al obtener los datos.");
                return response.json();
            })
            .then((data) => {
                loading.classList.add("hidden");

                if (!data || data.length === 0) {
                    noData.classList.remove("hidden");
                    return;
                }

                // Ordenar por cantidad de vuelos (descendente)
                data.sort((a, b) => b.flight_count - a.flight_count);

                // Filtrar top N si corresponde
                if (topN !== "all") {
                    data = data.slice(0, parseInt(topN));
                }

                const aerolineas = data.map((item) => item.icao_prefix);
                const vuelos = data.map((item) => item.flight_count);

                // Limpiar gráfica anterior si existe
                if (chart) {
                    chart.destroy();
                }

                const ctx = canvas.getContext("2d");
                chart = new Chart(ctx, {
                    type: "bar",
                    data: {
                        labels: aerolineas,
                        datasets: [{
                            label: "Número de Vuelos",
                            data: vuelos,
                            backgroundColor: "rgba(54, 162, 235, 0.6)",
                            borderColor: "rgba(54, 162, 235, 1)",
                            borderWidth: 1,
                        }],
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { display: false },
                            title: {
                                display: true,
                                text: `Top ${topN === "all" ? "Todos" : topN} Aerolíneas por Cantidad de Vuelos`,
                                font: { size: 18 },
                            },
                        },
                        scales: {
                            x: { title: { display: true, text: "Aerolínea (prefijo ICAO)" } },
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: "Cantidad de Vuelos" },
                            },
                        },
                    },
                });
            })
            .catch((error) => {
                loading.classList.add("hidden");
                noData.classList.remove("hidden");
                console.error("Error cargando los datos de aerolíneas:", error);
            });
    }

    // Evento para cambiar el topN
    topNSelect.addEventListener("change", () => {
        const value = topNSelect.value;
        fetchAndRender(value);
    });

    // Cargar por defecto
    fetchAndRender(topNSelect.value);



}

function renderAeropuertosChart() {
    fetch("/api/aeropuertos-frecuentes")
        .then(response => response.json())
        .then(data => {
            const salidaLabels = data.salidas.map(item => item.airport_name);
            const salidaCounts = data.salidas.map(item => item.flight_count);

            const llegadaLabels = data.llegadas.map(item => item.airport_name);
            const llegadaCounts = data.llegadas.map(item => item.flight_count);

            const ctxSalida = document.getElementById("aeropuertosSalidaChart").getContext("2d");
            new Chart(ctxSalida, {
                type: "bar",
                data: {
                    labels: salidaLabels,
                    datasets: [{
                        label: "Salidas",
                        data: salidaCounts,
                        backgroundColor: "rgba(75, 192, 192, 0.6)",
                        borderColor: "rgba(75, 192, 192, 1)",
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: "Aeropuertos de Salida más Frecuentes"
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: "Aeropuerto" } },
                        y: { beginAtZero: true, title: { display: true, text: "Número de Vuelos" } }
                    }
                }
            });

            const ctxLlegada = document.getElementById("aeropuertosDestinoChart").getContext("2d");
            new Chart(ctxLlegada, {
                type: "bar",
                data: {
                    labels: llegadaLabels,
                    datasets: [{
                        label: "Llegadas",
                        data: llegadaCounts,
                        backgroundColor: "rgba(255, 159, 64, 0.6)",
                        borderColor: "rgba(255, 159, 64, 1)",
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: "Aeropuertos de Destino más Frecuentes"
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: "Aeropuerto" } },
                        y: { beginAtZero: true, title: { display: true, text: "Número de Vuelos" } }
                    }
                }
            });
        })
        .catch(error => {
            console.error("Error al cargar datos de aeropuertos:", error);
        });
}

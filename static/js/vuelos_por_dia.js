document.addEventListener("DOMContentLoaded", () => {
    const yearSelect = document.getElementById("year");
    const monthSelect = document.getElementById("month");
    const daySelect = document.getElementById("day");
    const applyButton = document.getElementById("applyFilters");

    const chartHoraCtx = document.getElementById("vuelosPorDiaChart").getContext("2d");
    const chartMesCtx = document.getElementById("vuelosPorMesChart").getContext("2d");
    const chartDiaMesCtx = document.getElementById("vuelosPorDiaEnMesChart").getContext("2d");

    let chartHora;
    let chartMes;
    let chartDiaMes;

    async function loadDateOptions() {
        try {
            const response = await fetch("/api/fechas-disponibles");
            const data = await response.json();

            data.years.forEach(year => {
                const option = document.createElement("option");
                option.value = year;
                option.textContent = year;
                yearSelect.appendChild(option);
            });

            data.months.forEach(month => {
                const option = document.createElement("option");
                option.value = month;
                option.textContent = month.toString().padStart(2, '0');
                monthSelect.appendChild(option);
            });

            data.days.forEach(day => {
                const option = document.createElement("option");
                option.value = day;
                option.textContent = day.toString().padStart(2, '0');
                daySelect.appendChild(option);
            });
        } catch (error) {
            console.error("Error al cargar fechas:", error);
        }
    }

    async function loadChartData() {
        const year = yearSelect.value;
        const month = monthSelect.value;
        const day = daySelect.value;

        if (!year || !month || !day) {
            alert("Por favor selecciona año, mes y día.");
            return;
        }

        const url = new URL("/api/vuelos-por-hora/", window.location.origin);
        url.searchParams.append("year", year);
        url.searchParams.append("month", month);
        url.searchParams.append("day", day);

        try {
            const response = await fetch(url);
            const data = await response.json();

            if (chartHora) chartHora.destroy();

            const labels = data.map(item => `${item.hour}:00`);
            const counts = data.map(item => item.flight_count);

            chartHora = new Chart(chartHoraCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Vuelos por hora',
                        data: counts,
                        backgroundColor: 'rgba(59, 130, 246, 0.6)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Cantidad de Vuelos'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Hora'
                            }
                        }
                    }
                }
            });

        } catch (error) {
            console.error("Error al cargar el gráfico por hora:", error);
        }
    }

    async function loadMonthlyChartData() {
        const year = yearSelect.value;

        if (!year) return;

        const url = new URL("/api/vuelos-por-mes/", window.location.origin);
        url.searchParams.append("year", year);

        try {
            const response = await fetch(url);
            const data = await response.json();

            if (chartMes) chartMes.destroy();

            const labels = data.map(item => item.month);
            const counts = data.map(item => item.flight_count);

            chartMes = new Chart(chartMesCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Vuelos por mes',
                        data: counts,
                        backgroundColor: 'rgba(16, 185, 129, 0.6)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        fill: false,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Cantidad de Vuelos'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Mes'
                            }
                        }
                    }
                }
            });

            // También cargar el gráfico de vuelos por día en mes aquí
            loadDailyInMonthChartData(year, monthSelect.value);

        } catch (error) {
            console.error("Error al cargar el gráfico mensual:", error);
        }
    }

    async function loadDailyInMonthChartData(year, month) {
        if (!year || !month) return;

        const url = new URL("/api/vuelos-por-dia-en-mes/", window.location.origin);
        url.searchParams.append("year", year);
        url.searchParams.append("month", month);

        try {
            const response = await fetch(url);
            const data = await response.json();

            if (chartDiaMes) chartDiaMes.destroy();

            const labels = data.map(item => item.day.toString().padStart(2, '0'));
            const counts = data.map(item => item.flight_count);

            chartDiaMes = new Chart(chartDiaMesCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Vuelos por día en el mes',
                        data: counts,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Cantidad de Vuelos'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Día del mes'
                            }
                        }
                    }
                }
            });

        } catch (error) {
            console.error("Error al cargar gráfico de vuelos por día en mes:", error);
        }
    }

    applyButton.addEventListener("click", () => {
        loadChartData();
        loadDailyInMonthChartData(yearSelect.value, monthSelect.value);
    });

    yearSelect.addEventListener("change", () => {
        loadMonthlyChartData();
        loadDailyInMonthChartData(yearSelect.value, monthSelect.value);
    });

    monthSelect.addEventListener("change", () => {
        loadDailyInMonthChartData(yearSelect.value, monthSelect.value);
    });

    loadDateOptions().then(() => {
        if (yearSelect.value) {
            loadMonthlyChartData();
            loadDailyInMonthChartData(yearSelect.value, monthSelect.value);
        }
    });
});

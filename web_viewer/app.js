document.getElementById('csvFileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Mostramos un estado de carga por si el archivo es pesado
    document.getElementById('emgChart').innerHTML = '<p class="placeholder-text">Procesando datos...</p>';

    // Parseo del CSV
    Papa.parse(file, {
        header: true,
        dynamicTyping: true, // Convierte los números a float automáticamente
        skipEmptyLines: true,
        complete: function(results) {
            const data = results.data;
            generarGrafico(data);
        }
    });
});

function generarGrafico(data) {
    // Extraemos las columnas mapeando el array de resultados
    const time = data.map(row => row['Tiempo (s)']);
    const canal0 = data.map(row => row['Canal 0']);
    const canal1 = data.map(row => row['Canal 1']);
    const canal2 = data.map(row => row['Canal 2']);

    // Configuramos las 3 trazas (líneas)
    const trace0 = {
        x: time,
        y: canal0,
        mode: 'lines',
        name: 'Canal 0',
        line: { color: '#FF0000', width: 1.5 } // Rojo vibrante
    };

    const trace1 = {
        x: time,
        y: canal1,
        mode: 'lines',
        name: 'Canal 1',
        line: { color: '#FFFFFF', width: 1.5 } // Blanco
    };

    const trace2 = {
        x: time,
        y: canal2,
        mode: 'lines',
        name: 'Canal 2',
        line: { color: '#888888', width: 1.5 } // Gris oscuro
    };

    const layout = {
        title: 'Visualización de Señales EMG',
        plot_bgcolor: '#000000',
        paper_bgcolor: '#000000',
        font: { color: '#ffffff' },
        xaxis: {
            title: 'Tiempo (s)',
            gridcolor: '#333333',
            zerolinecolor: '#555555'
        },
        yaxis: {
            title: 'Amplitud (V)',
            gridcolor: '#333333',
            zerolinecolor: '#555555'
        },
        // Modo oscuro para el menú de herramientas (zoom, pan, etc)
        modebar: { color: '#ffffff', activecolor: '#FF0000' }
    };

    const config = {
        responsive: true,
        scrollZoom: true, // Permite hacer zoom con la ruedita del mouse
        displaylogo: false
    };

    // Renderizamos el gráfico
    Plotly.newPlot('emgChart', [trace0, trace1, trace2], layout, config);
}

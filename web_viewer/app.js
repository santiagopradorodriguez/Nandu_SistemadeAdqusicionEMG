// --- LÓGICA DE PESTAÑAS ---
function openTab(evt, tabName) {
    // Oculta todos los contenidos de las pestañas
    const tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove("active");
    }

    // Quita la clase "active" de todos los botones
    const tabBtns = document.getElementsByClassName("tab-btn");
    for (let i = 0; i < tabBtns.length; i++) {
        tabBtns[i].classList.remove("active");
    }

    // Muestra la pestaña actual y añade "active" al botón que se hizo clic
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// --- PESTAÑA 1: DATOS CRUDOS (CSV a Gráfico interactivo) ---
document.getElementById('csvFileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    document.getElementById('csvFileInfo').innerText = `Cargado: ${file.name}`;
    document.getElementById('emgChart').innerHTML = '<p class="placeholder-text" style="margin:auto;">Procesando datos...</p>';

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            generarGrafico(results.data);
        }
    });
});

function generarGrafico(data) {
    const time = data.map(row => row['Tiempo (s)']);
    const canal0 = data.map(row => row['Canal 0']);
    const canal1 = data.map(row => row['Canal 1']);
    const canal2 = data.map(row => row['Canal 2']);

    const trace0 = { x: time, y: canal0, mode: 'lines', name: 'Canal 0', line: { color: '#FF0000', width: 1.5 } };
    const trace1 = { x: time, y: canal1, mode: 'lines', name: 'Canal 1', line: { color: '#FFFFFF', width: 1.5 } };
    const trace2 = { x: time, y: canal2, mode: 'lines', name: 'Canal 2', line: { color: '#888888', width: 1.5 } };

    const layout = {
        plot_bgcolor: '#0a0a0a',
        paper_bgcolor: '#0a0a0a',
        font: { color: '#ffffff' },
        margin: { t: 30, b: 40, l: 50, r: 20 },
        xaxis: { title: 'Tiempo (s)', gridcolor: '#333333', zerolinecolor: '#555555' },
        yaxis: { title: 'Amplitud', gridcolor: '#333333', zerolinecolor: '#555555' },
        modebar: { color: '#ffffff', activecolor: '#FF0000' }
    };

    const config = {
        responsive: true,
        scrollZoom: true, // Esto reemplaza todo el código de sliders de Matplotlib
        displaylogo: false
    };

    Plotly.newPlot('emgChart', [trace0, trace1, trace2], layout, config);
}

// --- PESTAÑA 2: COMPARADOR (Galería de PNGs) ---
document.getElementById('pngFileInput').addEventListener('change', function(e) {
    const files = e.target.files;
    if (files.length === 0) return;

    const gallery = document.getElementById('imageGallery');
    gallery.innerHTML = ''; // Limpia la galería

    // Itera sobre todos los archivos de imagen seleccionados
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Creamos un objeto URL temporal en la memoria del navegador
        const imageURL = URL.createObjectURL(file);
        
        // Creamos el elemento imagen y lo agregamos a la grilla
        const imgElement = document.createElement('img');
        imgElement.src = imageURL;
        imgElement.title = file.name; // Al pasar el mouse muestra el nombre del archivo
        
        gallery.appendChild(imgElement);
    }
});
// Diccionario de la base de datos
let databaseFiles = {};

// Variables globales para el procesamiento de señales
let datosCrudosParseados = null;
let notchQ = 5.0; // Factor Q por defecto para el Notch (violencia)

document.getElementById('folderInput').addEventListener('change', function(e) {
    const files = e.target.files;
    if (files.length === 0) return;

    databaseFiles = {};

    // LÓGICA ROBUSTA PARA LEER CARPETAS
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const pathParts = file.webkitRelativePath.split('/');
        
        // Buscamos específicamente grabacion.csv sin importar qué tan profundo esté
        if (file.name === 'grabacion.csv') {
            // El nombre de la carpeta de medición es el padre inmediato del csv
            const folderName = pathParts[pathParts.length - 2];
            if (!databaseFiles[folderName]) databaseFiles[folderName] = { csv: null, metadata: null };
            databaseFiles[folderName].csv = file;
        }
        
        // Buscamos específicamente metadata.json dentro de la carpeta canal_0
        if (file.name === 'metadata.json' && file.webkitRelativePath.includes('canal_0')) {
            // El abuelo de metadata.json (canal_0 -> Medicion) es el nombre de la carpeta
            const folderName = pathParts[pathParts.length - 3];
            if (!databaseFiles[folderName]) databaseFiles[folderName] = { csv: null, metadata: null };
            databaseFiles[folderName].metadata = file;
        }
    }

    const select = document.getElementById('measurementSelect');
    select.innerHTML = '<option value="">Seleccione una medición</option>';
    
    // Filtrar para mostrar solo las que tengan un CSV
    const validFolders = Object.keys(databaseFiles).filter(folder => databaseFiles[folder].csv !== null);
    
    validFolders.forEach(folder => {
        const option = document.createElement('option');
        option.value = folder;
        option.innerText = folder;
        select.appendChild(option);
    });

    // Mostrar el menú desplegable
    select.style.display = 'inline-block';
    document.getElementById('selectedFolderName').innerText = `(${validFolders.length} encontradas)`;
});

// Evento al seleccionar del menú
document.getElementById('measurementSelect').addEventListener('change', function(e) {
    const selectedFolder = e.target.value;
    const infoText = document.getElementById('selectedFolderName');
    const placeholder = document.getElementById('placeholderText');
    const chartDiv = document.getElementById('emgChart');
    const toolsPanel = document.getElementById('toolsPanel');
    
    if (!selectedFolder) return;

    infoText.innerText = selectedFolder;
    
    // Cambiar la vista inicial: Ocultar texto, mostrar gráfico y herramientas
    placeholder.innerText = 'Cargando datos...';
    chartDiv.style.display = 'none';
    
    const targetCSV = databaseFiles[selectedFolder].csv;
    const targetMeta = databaseFiles[selectedFolder].metadata;

    // Cargar Metadata
    const metaDisplay = document.getElementById('metadataDisplay');
    if (targetMeta) {
        const reader = new FileReader();
        reader.onload = function(evt) {
            try {
                const jsonObj = JSON.parse(evt.target.result);
                metaDisplay.innerText = JSON.stringify(jsonObj, null, 2);
            } catch (err) {
                metaDisplay.innerText = "Error leyendo JSON.";
            }
        };
        reader.readAsText(targetMeta);
    } else {
        metaDisplay.innerText = "No se encontró metadata.json\nen canal_0.";
    }

    // Parsear el CSV
    Papa.parse(targetCSV, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            // Una vez parseado, revelamos el gráfico y el panel lateral
            placeholder.style.display = 'none';
            chartDiv.style.display = 'block';
            toolsPanel.style.display = 'flex'; 
            
            // Guardamos los datos crudos y disparamos el procesado
            datosCrudosParseados = results.data;
            procesarYGraficar();
        }
    });
});

// ==========================================
// LÓGICA DE PROCESAMIENTO DSP Y GRAFICACIÓN
// ==========================================

// Escuchamos si el usuario toca las casillas o cambia las frecuencias
document.getElementById('chkNotch').addEventListener('change', procesarYGraficar);
document.getElementById('freqNotch').addEventListener('input', procesarYGraficar);
document.getElementById('chkBP').addEventListener('change', procesarYGraficar);
document.getElementById('freqLow').addEventListener('input', procesarYGraficar);
document.getElementById('freqHigh').addEventListener('input', procesarYGraficar);

function procesarYGraficar() {
    if (!datosCrudosParseados) return;

    // Extraemos los arrays
    let time = datosCrudosParseados.map(row => row['Tiempo (s)']);
    let c0 = datosCrudosParseados.map(row => row['Canal 0']);
    let c1 = datosCrudosParseados.map(row => row['Canal 1']);
    let c2 = datosCrudosParseados.map(row => row['Canal 2']);

    // Limpieza de seguridad: Si la última fila del CSV está vacía, la borramos
    if (time[time.length - 1] === null || time[time.length - 1] === undefined) {
        time.pop(); c0.pop(); c1.pop(); c2.pop();
    }

    // Calculamos la frecuencia de muestreo real
    let fs = 1 / (time[1] - time[0]);

    const aplicarNotch = document.getElementById('chkNotch').checked;
    const aplicarPB = document.getElementById('chkBP').checked;

    var iirCalculator = new Fili.CalcCascades();

    // 1. EL PASABANDA Y EL OFFSET
    if (aplicarPB) {
        const fLow = parseFloat(document.getElementById('freqLow').value);
        const fHigh = parseFloat(document.getElementById('freqHigh').value);

        // Sacamos el offset ANTES del filtro para centrar en 0 y evitar el "latigazo"
        c0 = removerOffset(c0);
        c1 = removerOffset(c1);
        c2 = removerOffset(c2);

        // Creamos un Pasa-altos (Highpass)
        var hpCoeffs = iirCalculator.highpass({
            order: 2, // Orden 2 da una caída suave y estable
            characteristic: 'butterworth',
            Fs: fs,
            Fc: fLow,
            preGain: false
        });
        
        // Creamos un Pasa-bajos (Lowpass)
        var lpCoeffs = iirCalculator.lowpass({
            order: 2,
            characteristic: 'butterworth',
            Fs: fs,
            Fc: fHigh,
            preGain: false
        });

        var hpFilter0 = new Fili.IirFilter(hpCoeffs);
        var hpFilter1 = new Fili.IirFilter(hpCoeffs);
        var hpFilter2 = new Fili.IirFilter(hpCoeffs);

        var lpFilter0 = new Fili.IirFilter(lpCoeffs);
        var lpFilter1 = new Fili.IirFilter(lpCoeffs);
        var lpFilter2 = new Fili.IirFilter(lpCoeffs);

        // Aplicamos la cascada: la señal pasa primero por el HP y luego por el LP
        c0 = lpFilter0.multiStep(hpFilter0.multiStep(c0));
        c1 = lpFilter1.multiStep(hpFilter1.multiStep(c1));
        c2 = lpFilter2.multiStep(hpFilter2.multiStep(c2));
    }

    // 2. EL NOTCH (Rechaza-banda)
    if (aplicarNotch) {
        const fNotch = parseFloat(document.getElementById('freqNotch').value);
        
        // Calculamos el Ancho de Banda (BW) a partir del Factor Q de "violencia"
        const bandwidth = fNotch / notchQ; 

        // Fili.js IIR usa Fc y BW, no F1 y F2
        var notchCoeffs = iirCalculator.bandstop({
            order: 1, // Orden 1 es un "biquad Notch" clásico. 
            characteristic: 'butterworth',
            Fs: fs,
            Fc: fNotch,
            BW: bandwidth,
            preGain: false
        });

        var notchFilter0 = new Fili.IirFilter(notchCoeffs);
        var notchFilter1 = new Fili.IirFilter(notchCoeffs);
        var notchFilter2 = new Fili.IirFilter(notchCoeffs);

        c0 = notchFilter0.multiStep(c0);
        c1 = notchFilter1.multiStep(c1);
        c2 = notchFilter2.multiStep(c2);
    }

    // 3. GRAFICAMOS
    const trace0 = { x: time, y: c0, mode: 'lines', name: 'Canal 0', line: { color: '#FF0000', width: 1.5 } };
    const trace1 = { x: time, y: c1, mode: 'lines', name: 'Canal 1', line: { color: '#FFFFFF', width: 1.5 } };
    const trace2 = { x: time, y: c2, mode: 'lines', name: 'Canal 2', line: { color: '#888888', width: 1.5 } };

    const layout = {
        plot_bgcolor: '#0a0a0a',
        paper_bgcolor: '#0a0a0a',
        font: { color: '#ffffff' },
        margin: { t: 30, b: 40, l: 50, r: 20 },
        xaxis: { title: 'Tiempo (s)', gridcolor: '#333333', zerolinecolor: '#555555' },
        yaxis: { title: 'Amplitud', gridcolor: '#333333', zerolinecolor: '#555555' },
        modebar: { color: '#ffffff', activecolor: '#FF0000' }
    };

    const config = { responsive: true, scrollZoom: true, displaylogo: false };
    Plotly.react('emgChart', [trace0, trace1, trace2], layout, config);
}

// Función auxiliar para quitar el valor medio (offset = 0)
function removerOffset(array) {
    let suma = 0;
    for(let i = 0; i < array.length; i++) {
        suma += array[i];
    }
    let media = suma / array.length;
    
    let sinOffset = new Array(array.length);
    for(let i = 0; i < array.length; i++) {
        sinOffset[i] = array[i] - media;
    }
    return sinOffset;
}

// ==========================================
// HERRAMIENTAS ADICIONALES Y PESTAÑAS
// ==========================================

// Botón de Exportar
document.getElementById('btnExport').addEventListener('click', function() {
    const folder = document.getElementById('measurementSelect').value;
    let fNotch = document.getElementById('chkNotch').checked ? "_Notch" : "";
    let fPB = document.getElementById('chkBP').checked ? "_PB" : "";
    
    const fileName = `${folder}_Canales-Visualizados${fNotch}${fPB}`;

    Plotly.downloadImage('emgChart', {
        format: 'png',
        filename: fileName,
        height: 600,
        width: 1000
    });
});

// Configuración de Pendiente Notch (Q)
function configurarNotch() {
    const inputQ = prompt("Configuración del filtro Notch\nIngresá el Factor Q (Mayor Q = Corte más fino/violento. Sugerido: entre 2 y 50):", notchQ);
    if(inputQ !== null && !isNaN(parseFloat(inputQ))) {
        notchQ = parseFloat(inputQ);
        console.log("Factor Q seteado a:", notchQ);
        // Recalculamos al instante si la casilla estaba tildada
        if(document.getElementById('chkNotch').checked) {
            procesarYGraficar();
        }
    }
}

function configurarPasabanda() {
    alert("Próximamente: Configurar orden del filtro (ej. 2, 4, 6) para definir qué tan abrupta es la caída de los bordes.");
}

// Pestañas
function openTab(evt, tabName) {
    const tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) tabContents[i].classList.remove("active");
    const tabBtns = document.getElementsByClassName("tab-btn");
    for (let i = 0; i < tabBtns.length; i++) tabBtns[i].classList.remove("active");
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Lógica de Pestaña Comparador
document.getElementById('pngFileInput').addEventListener('change', function(e) {
    const files = e.target.files;
    if (files.length === 0) return;
    const gallery = document.getElementById('imageGallery');
    gallery.innerHTML = '';
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const imageURL = URL.createObjectURL(file);
        const imgElement = document.createElement('img');
        imgElement.src = imageURL;
        imgElement.title = file.name;
        gallery.appendChild(imgElement);
    }
});
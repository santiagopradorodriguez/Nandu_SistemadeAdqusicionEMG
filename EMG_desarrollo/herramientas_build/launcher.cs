using System;
using System.Drawing;
using System.Windows.Forms;
using System.Diagnostics;
using System.Threading;
using System.IO;
using System.Text;

namespace NanduLsdLauncher
{
    public class SplashScreen : Form
    {
        private System.Windows.Forms.Timer pollTimer;
        private Process coreProcess;

        public SplashScreen()
        {
            this.FormBorderStyle = FormBorderStyle.None;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.Size = new Size(400, 250);
            this.BackColor = Color.FromArgb(10, 10, 15);
            
            Label lblTitle = new Label();
            lblTitle.Text = "Nandu LSD";
            lblTitle.ForeColor = Color.Cyan;
            lblTitle.Font = new Font("Courier New", 24, FontStyle.Bold);
            lblTitle.AutoSize = false;
            lblTitle.TextAlign = ContentAlignment.MiddleCenter;
            lblTitle.Dock = DockStyle.Top;
            lblTitle.Height = 120;
            
            Label lblSub = new Label();
            lblSub.Text = "Iniciando motor DSP y cargando Python...";
            lblSub.ForeColor = Color.White;
            lblSub.Font = new Font("Segoe UI", 10, FontStyle.Regular);
            lblSub.AutoSize = false;
            lblSub.TextAlign = ContentAlignment.MiddleCenter;
            lblSub.Dock = DockStyle.Fill;
            
            this.Controls.Add(lblSub);
            this.Controls.Add(lblTitle);

            try {
                this.Icon = new Icon("icono.ico");
            } catch { }

            this.Load += OnLoad;
        }

        private void OnLoad(object sender, EventArgs e)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                string exePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "NanduLsd_Core.exe");
                if (!File.Exists(exePath))
                {
                    exePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "NanduLsd_Core", "NanduLsd_Core.exe");
                }
                psi.FileName = exePath;
                psi.UseShellExecute = false;
                
                coreProcess = Process.Start(psi);
                
                pollTimer = new System.Windows.Forms.Timer();
                pollTimer.Interval = 200;
                pollTimer.Tick += CheckIfCoreIsReady;
                pollTimer.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error iniciando el motor principal: " + ex.Message, "Error Critico", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Application.Exit();
            }
        }

        private void CheckIfCoreIsReady(object sender, EventArgs e)
        {
            if (coreProcess == null || coreProcess.HasExited)
            {
                Application.Exit();
                return;
            }
            
            coreProcess.Refresh();
            if (coreProcess.MainWindowHandle != IntPtr.Zero)
            {
                pollTimer.Stop();
                Thread.Sleep(800); 
                Application.Exit();
            }
        }

        [STAThread]
        public static int Main(string[] args)
        {
            if (args != null && args.Length > 0)
            {
                try
                {
                    ProcessStartInfo psi = new ProcessStartInfo();
                    string exePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "NanduLsd_Core.exe");
                    if (!File.Exists(exePath))
                    {
                        exePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "NanduLsd_Core", "NanduLsd_Core.exe");
                    }
                    psi.FileName = exePath;
                    psi.UseShellExecute = false;

                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < args.Length; i++)
                    {
                        if (i > 0) sb.Append(" ");
                        if (args[i].Contains(" ") || args[i].Contains("\t"))
                            sb.Append("\"").Append(args[i]).Append("\"");
                        else
                            sb.Append(args[i]);
                    }
                    psi.Arguments = sb.ToString();

                    using (Process p = Process.Start(psi))
                    {
                        p.WaitForExit();
                        return p.ExitCode;
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error ejecutando script: " + ex.Message, "Error Critico", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return 1;
                }
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SplashScreen());
            return 0;
        }
    }
}

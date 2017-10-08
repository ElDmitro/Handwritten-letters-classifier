using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.Net;

namespace Handwritten_letters_classifier
{
    public partial class Form1 : Form
    {


        Point p1;
        string path_to_image = @"image.jpg";
        string path_to_tf_server = @"http://localhost:5000/";

        WebClient wc = new WebClient();

        static ImageCodecInfo jpgEncoder = GetEncoder(ImageFormat.Jpeg);
        static Encoder enc = Encoder.Quality;
        EncoderParameter eParameter = new EncoderParameter(enc, 100L);
        EncoderParameters eParameters = new EncoderParameters(1);
        
        public Form1()
        {
            InitializeComponent();
            pictureBox1.Size = new Size(280, 280);
            textBox2.Size = new Size(280, 280);
            textBox2.Location = pictureBox1.Location;
            textBox2.Text = "Загрузка...";
            pictureBox1.Image = (Image)new Bitmap(28, 28);
            eParameters.Param[0] = eParameter;
        }


        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                Point p2 = new Point((int)Math.Round(e.X / 10.0), (int)Math.Round(e.Y / 10.0));
                Bitmap im = (Bitmap)pictureBox1.Image;
                using (Graphics graphics = Graphics.FromImage(im))
                {
                    graphics.DrawLine(new Pen(Color.Black, 3), p1, p2);
                }
                pictureBox1.Image = im;
                p1 = p2;
            }
        }
        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            p1 = new Point((int)Math.Round(e.X / 10.0), (int)Math.Round(e.Y / 10.0));
        }
        private void button1_Click(object sender, EventArgs e)
        {
            pictureBox1.Image = (Image)new Bitmap(28, 28);
            textBox1.Text = "";
            pictureBox1.Enabled = true;
        }

        private void button2_Click(object sender, EventArgs e)

        {
            pictureBox1.Enabled = false;
            Bitmap savedBit = new Bitmap(28, 28);
            pictureBox1.Width = 28;
            pictureBox1.Height = 28;


            pictureBox1.DrawToBitmap(savedBit, pictureBox1.ClientRectangle);
            pictureBox1.Width = 280;
            pictureBox1.Height = 280;
            savedBit.Save(path_to_image, jpgEncoder, eParameters);

            try
            {
                textBox1.Text = wc.DownloadString(path_to_tf_server + path_to_image);
            }
            catch(System.Net.WebException)
            {
                MessageBox.Show("Ошибка подключения.\r\n Запустите python-server из Debug каталога проекта и повторите попытку.");

            }
            
        }
        private static ImageCodecInfo GetEncoder(ImageFormat f)
        {

            ImageCodecInfo[] codecList = ImageCodecInfo.GetImageDecoders();

            foreach (ImageCodecInfo codec in codecList)
            {
                if (codec.FormatID == f.Guid)
                {
                    return codec;
                }
            }
            return null;
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
    }
}

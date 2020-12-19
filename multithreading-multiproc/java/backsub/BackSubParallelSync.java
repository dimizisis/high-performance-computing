
import java.util.Random;

class BackSubParallelSync{

    static final int RAND_MAX = 0x7fff;

    public static void main(String[] args) throws InterruptedException{

        if (args.length != 1) {
            System.out.printf("Usage : BackSubParallelSync <matrix size>\n");
            System.exit(1);
        }

        int N = Integer.parseInt(args[0]);

        double[] x = new double[N]; 
        double[] b = new double[N]; 
        double[][] a = new double[N][N];

        /* Create doubles between 0 and 1. Diagonal elents between 2 and 3. */
        for (int i = 0; i < N; i++) {
            x[i] = 0.0;
            b[i] = new Random().nextDouble()/(RAND_MAX*2.0-1.0);
            a[i][i] = 2.0+new Random().nextDouble()/(RAND_MAX*2.0-1.0);
            for (int j = 0; j < i; j++) 
                a[i][j] = new Random().nextDouble()/(RAND_MAX*2.0-1.0);
        }

        BackSubThread[] threads = new BackSubThread[N];

        /* Create threads */
        for(int i=0; i<N; i++)
           threads[i] = new BackSubThread(x, a, b, i);

        /* Start threads */
        for(int i=0; i<N; i++)
            threads[i].start();

        /* Wait for threads to finish */
        for(int i=0; i<N; i++)
            threads[i].join();
        
        /* Print result */
        for (int i = 0; i < N; i++)
            System.out.println(x[i]);
        
    }

    private static class BackSubThread extends Thread {

        private double[] x, b;
        private double[][] a;
        private int i;

        public BackSubThread(double[] x, double[][] a, double[] b, int i){
            this.x = x;
            this.a = a;
            this.b = b;
            this.i = i;
        }

        @Override
        public void run(){
            double sum = 0.0;
            for (int j = 0; j < i; j++){
                synchronized(this.getClass()){
                    sum = sum + (x[j] * a[i][j]);
                }
            }
            synchronized(this.getClass()){
                x[i] = (b[i] - sum) / a[i][i];
            }
        }
    }

}
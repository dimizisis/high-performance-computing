
import java.util.Random;

class BackSubParallelStaticBlocks{

    static final int RAND_MAX = 0x7fff;

    public static void main(String[] args) throws InterruptedException{

        if (args.length != 2) {
            System.out.printf("Usage : BackSubParallelBlocks <matrix size> <num_threads>\n");
            System.exit(1);
        }

        int N = Integer.parseInt(args[0]);
        int numThreads = Integer.parseInt(args[1]);

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

        /* Create sufficient blocks */
        int blocks = N / numThreads;

        BackSubThread[] threads = new BackSubThread[numThreads];

        /* Create threads */
        for(int i=0, j=0; i<numThreads; ++i, j+=blocks)
            threads[i] = new BackSubThread(x, a, b, j, i, blocks);

        /* Start threads */
        for(int i=0; i<numThreads; i++)
            threads[i].start();

        /* Wait for threads to finish */
        for(int i=0; i<numThreads; i++)
            threads[i].join();
        
        /* Print result */
        for (int i = 0; i < N; i++)
            System.out.println(x[i]);
        
    }

    private static class BackSubThread extends Thread {

        private double[] x, b;
        private double[][] a;
        private int blocks;
        private int idx;
        private int rank;

        public BackSubThread(double[] x, double[][] a, double[] b, int idx, int rank, int blocks){
            this.x = x;
            this.a = a;
            this.b = b;
            this.idx = idx;
            this.rank = rank;
            this.blocks = blocks;
        }

        @Override
        public void run(){
            double sum;
            int stop = rank*blocks + blocks;
            for (int i=idx; i<stop; ++i){
                sum = 0.0;
                for (int j = 0; j < i; j++)
                    sum = sum + (x[j] * a[i][j]);
                x[i] = (b[i] - sum) / a[i][i];
            }
        }
    }

}
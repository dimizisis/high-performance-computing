import java.io.RandomAccessFile;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

class CharFreqSeq {

	static final int N = 128;
    static final int base = 0;
	public static void main(String[] args) throws IOException {

		RandomAccessFile pFile;
		long fileSize;
		char[] buffer;
		String filename;
		int[] freq = new int[N];
		long start, end;

		if(args.length != 1) {
			System.out.println("Usage : CharFreqSeq <file_name>");
			System.exit(1);
			return;
        }
        
        filename = args[0];
        
		try {
			pFile = new RandomAccessFile(filename.toString(), "r");
		} catch(IOException ex) {
			System.out.println("File error");
			System.exit(2);
			return;
		}

		// obtain file size:
		pFile.seek(pFile.length());
		fileSize = pFile.getFilePointer();
		pFile.seek(0);
		System.out.println("file size is " + fileSize);

		byte[] byteArray = new byte[(int)fileSize];
		pFile.readFully(byteArray);

        // allocate memory to contain the file:
        buffer = new String(byteArray, StandardCharsets.UTF_8).toCharArray();

        zeros(freq, N);

		start = System.currentTimeMillis();

		countCharacters(freq, buffer, fileSize);

		end = System.currentTimeMillis();

		displayCount(freq, N);

		System.out.printf("Time spent for counting: %g", (double)(end - start) / 1000);

		pFile.close();
    }

    public static void displayCount(int[] freq, int n) {
		for(int j = 0; j < n; ++j) {
			System.out.println(j + " = " + freq[j]);
		}
	}

    private static void zeros(int[] array, int n){
        for (int j=0; j<n; ++j)
            array[j]=0;
    }
    
    private static void countCharacters(int[] freq, char[] buffer, long fileSize){
        for(int i = 0; i < fileSize; ++i) {
			++freq[buffer[i] - base];
		}
	}
}
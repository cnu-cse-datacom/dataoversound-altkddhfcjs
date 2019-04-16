package com.example.sound.devicesound;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import calsualcoding.reedsolomon.EncoderDecoder;
import google.zxing.common.reedsolomon.ReedSolomonException;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

import java.util.ArrayList;
import java.util.List;

/** 201402340 Kim Hyunseop
 *  DataCommunity Homework05
 *  Pied Piper**/

public class Listentone {

    int HANDSHAKE_START_HZ = 4096;
    int HANDSHAKE_END_HZ = 5120 + 1024;

    int START_HZ = 1024;
    int STEP_HZ = 256;
    int BITS = 4;

    int FEC_BYTES = 4;

    private int mAudioSource = MediaRecorder.AudioSource.MIC;
    private int mSampleRate = 44100;
    private int mChannelCount = AudioFormat.CHANNEL_IN_MONO;
    private int mAudioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private float interval = 0.1f;

    private int mBufferSize = AudioRecord.getMinBufferSize(mSampleRate, mChannelCount, mAudioFormat);

    public AudioRecord mAudioRecord = null;
    int audioEncodig;
    /*in Packet*/
    boolean startFlag;
    FastFourierTransformer transform;
    /* Using reedsolomon package */
    EncoderDecoder ed;

    public Listentone(){

        transform = new FastFourierTransformer(DftNormalization.STANDARD);
        startFlag = false;
        mAudioRecord = new AudioRecord(mAudioSource, mSampleRate, mChannelCount, mAudioFormat, mBufferSize);
        mAudioRecord.startRecording();
        ed = new EncoderDecoder();
    }

    /*https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html*/
    /*https://github.com/numpy/numpy/blob/v1.16.1/numpy/fft/helper.py#L131-L177*/
    /*   Parameters
     *   ----------
     *   n : int
     *       Window length.
     *   d : scalar, optional
     *       Sample spacing (inverse of the sampling rate). Defaults to 1.
     *   Returns
     *   -------
     *   f : ndarray
     *   Array of length `n` containing the sample frequencies.*/
    /* def fftfreq(n, d=1.0):
     *    if not isinstance(n, integer_types):
     *    raise ValueError("n should be an integer")
     *    val = 1.0 / (n * d)
     *    results = empty(n, int)
     *    N = (n-1)//2 + 1
     *    p1 = arange(0, N, dtype=int)
     *    results[:N] = p1
     *    p2 = arange(-(n//2), 0, dtype=int)
     *    results[N:] = p2
     *    return results * val
     *
     * */
    // n = length
    private Double[] fftfreq(int length, int duration){
        //Log.d("ListenToneFFTLength: ", String.valueOf(length));
        //Log.d("ListenToneTest1: ", "FFT");
        double val = 1.0 / (length * duration);
        //results = empty(n, int)
        int[] result = new int[length];
        int N = (length - 1) / 2 + 1;
        /* p1 = arange(0, N, dtype=int) */
        int[] p1 = new int[N];
        for(int i = 0; i < N; i++){
            p1[i] = i;
        }
        /* result[:N] = p1 */
        //System.arraycopy(p1, 0, result, 0, N);

        for(int i = 0; i < N; i ++){
            result[i] = p1[i];
        }
        /* p2 = arange(-(n//2), 0, dtype=int) */
        int[] p2 = new int[N];
        for(int i = 0; i < N; i++){
            p2[i] = -1 * (length / 2) + i;
        }
        /* result[N:] = p2 */
        //System.arraycopy(p2, 0, result, N, N*2);
        for(int i = 0; i < N; i ++){
            result[N+i] = p2[i];
        }
        /*return results * val*/
        Double[] results = new Double[length];
        for(int i = 0; i < length; i++){
            results[i] = result[i]*val;
        }

        return results;
    }

    /*dominant(frame_rate, chunk)*/
    private double findFrequency(short[] toTransform){
        int len = findPowerSize(toTransform.length);
        //Log.d("ListenToneInLen: ", String.valueOf(len));
        double[] real = new double[len];
        double[] img = new double[len];
        double realNum;
        double imgNum;
        double[] mag = new double[len];
        //Reed solomon - zero padding
        double[] DoubletoTranform = new double[len];

        for(int i = 0; i < toTransform.length; i++) {
            DoubletoTranform[i] = (double) toTransform[i];
        }
        for(int i = toTransform.length; i < len; i++){
            DoubletoTranform[i] = 0;
        }

        /* freqs = np.fft.fftfreq(len(chunk)) */
        Complex[] complx = transform.transform(DoubletoTranform, TransformType.FORWARD);
        //Log.d("ListenToneComplxSize: ", String.valueOf(complx.length));
        Double[] freq = this.fftfreq(complx.length, 1);

        for(int i = 0; i < complx.length; i++){
            realNum = complx[i].getReal();
            imgNum = complx[i].getImaginary();
            mag[i] = Math.sqrt((realNum * realNum) + (imgNum * imgNum));
        }

        /* peak_coeff = np.argmax(np.abs(w)) */
        double peak_coeff = 0;
        int peak_coeff_index = 0;

        for(int i = 0; i < complx.length; i++){
            if(peak_coeff < mag[i]){
                peak_coeff_index = i;
                peak_coeff = mag[i];
            }
        }
        return Math.abs(freq[peak_coeff_index]*mSampleRate); //in Hz
    }

    /*decode bitchunks*/
    private byte[] decode_bitchunks(int chunk_bits, List<Integer> chunks){
        //Log.d("ListenToneInChunks: ", String.valueOf(chunks));
        List<Integer> out_byte = new ArrayList<>();
        int next_read_chunk = 0;
        int next_read_bit = 0;
        int Byte = 0;
        int bits_left = 8;

        while (next_read_chunk < chunks.size()){
            int can_fill = chunk_bits - next_read_bit;
            int to_fill = Math.min(bits_left, can_fill);
            int offset = chunk_bits - next_read_bit - to_fill;
            // Log.d("ListenToneTest2: ", "Test2");
            Byte <<= to_fill;
            int shifted = chunks.get(next_read_chunk) & (((1 << to_fill) - 1) << offset);
            Byte |= shifted >> offset;

            bits_left -= to_fill;
            next_read_bit += to_fill;

            if(bits_left <= 0){
                out_byte.add(Byte);
                Byte = 0;
                bits_left = 8;
            }

            if(next_read_bit >= chunk_bits){
                next_read_chunk += 1;
                next_read_bit -= chunk_bits;
            }
        }

        //Log.d("ListenToneOutByte: ", String.valueOf(out_byte));
        /* convert integer to byte */
        byte[] out_bytes = new byte[out_byte.size()];
        for(int i = 0; i < out_bytes.length; i++){
            out_bytes[i] = (out_byte.get(i)).byteValue();
        }
        return out_bytes;
    }

    /*extract packet*/
    private byte[] extract_packet(List<Double> freqs){
        Log.d("ListenToneFreqency: ", String.valueOf(freqs));
        /*freqs = freqs[::2]*/
        List<Double> temp_freqs = new ArrayList<>();
        for(int i = 0; i < freqs.size(); i = i + 2){
            temp_freqs.add(freqs.get(i));
        }
        /*bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]*/
        List<Integer> chunks = new ArrayList<>();
        for(int i = 0; i < temp_freqs.size(); i++){
            chunks.add((int)(Math.round((temp_freqs.get(i) - START_HZ) / STEP_HZ)));
        }
        //Log.d("ListenToneBitChunks: ", String.valueOf(chunks));

        /*bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]*/
        List<Integer> bit_chunks = new ArrayList<>();
        for(int i = 1; i < chunks.size(); i++){
            if( 0 <= chunks.get(i) && chunks.get(i) < (int)Math.pow(2, BITS)){
                bit_chunks.add(chunks.get(i));
            }
        }
        Log.d("ListenToneBitChunks: ", String.valueOf(bit_chunks));
        /*return bytearray(decode_bitchunks(BITS, bit_chunks))*/
        return decode_bitchunks(BITS, bit_chunks);

    }


    private int findPowerSize(int buffersize){
        int result = 1;
        for(int i = 1; result < buffersize; i++){
            result = (int)Math.pow(2, i);
        }
        //Log.d("ListenToneSize: ", String.valueOf(result));
        return result;
    }

    /*Listen Linux*/
    public void PreRequest() {
        /*packet*/
        List<Double> packet = new ArrayList<>();
        Log.d("ListenToneStartListen: ", "----Listen----");
        /* num_frames = int(round((interval / 2)*frame_rate)) */
        /* Reed solomon -> blocksize => 2205 */
        int blocksize = ((int)(interval / 2 * mSampleRate));
        Log.d("ListenToneBlockSize: ", String.valueOf(blocksize));
        /*mic.setperiodsize(num_frames)
         * Sets the actual period size in frames.*/
        short[] buffer = new short[blocksize];

        while (true) {
            /*  read(short[] audioData, int offsetInShorts, int sizeInShorts)
             *  Reads audio data from the audio hardware for recording into a short array.
             *  https://developer.android.com/reference/android/media/AudioRecord  */
            int bufferedReadResult = mAudioRecord.read(buffer, 0, blocksize);
            if (bufferedReadResult != blocksize) {
                Log.d("ListenToneContinue: ", "---continue----");
                continue;
            }
            //System.arraycopy((double)buffer, 0, transform, 0, blocksize);
            /*dom = dominant(frame_rate, chunk)*/

            double dom = findFrequency(buffer);
            //Log.d("ListenToneDom: ", String.valueOf(dom));
            /*End Input*/
            if (startFlag && Math.abs(dom - HANDSHAKE_END_HZ) < 20) {
                Log.d("ListenToneEnd: ", "----END_HZ----");
                Log.d("ListenTonePacketSize: ", String.valueOf(packet.size()));
                byte[] byte_stream = extract_packet(packet);
                try {
                    /* byte_stream = RSCodec(FEC_BYTES).decode(byte_stream) */
                    ed.decodeData(byte_stream, FEC_BYTES);
                    Log.d("ListenToneByte_stream: ", String.valueOf(byte_stream));

                    /*byte_stream = byte_stream.decode("utf-8")*/
                    String display = "";
                    for(int i = 0; i < byte_stream.length; i++){
                        //Only Accept ASCII CODE
                        int ASCII = (int) byte_stream[i];
                        if(0 < ASCII && ASCII <= 128){
                            display += (char)ASCII;
                        }
                    }
                    /*String result = "";

                    for (int i = 0; i < byte_stream.length; i++) {
                        result = result + Character.toString((char) ((int) byte_stream[i]));
                    }

                    String display = null;
                    try {
                        display = new String(result.getBytes("UTF-8"), "ISO-8859-1");
                    } catch (java.io.UnsupportedEncodingException e) {}

                    */
                    Log.d("ListenToneResult: ", display);

                } catch (ReedSolomonException e) {
                } catch (EncoderDecoder.DataTooLargeException e) {
                }

                packet.clear();
                startFlag = false;
            } else if (startFlag) {
                Log.d("ListenToneDom: ", String.valueOf(dom));
                packet.add(dom);
                /*Start Input*/
            } else if (Math.abs(dom - HANDSHAKE_START_HZ) < 20) {
                Log.d("ListenToneStart: ", "----START_HZ----");
                startFlag = true;
            }
        }
    }

}

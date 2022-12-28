import org.mnn.*;

public class InterpreterTest {
    public InterpreterTest() {
        System.loadLibrary("mnn_java");
    }
    public void run(String model) {
        System.out.println("load model: " + model);
        Interpreter interpreter = Interpreter.createFromFile(model);
        interpreter.setSessionMode(Interpreter.SessionMode.Session_Debug);
        interpreter.setCacheFile(".cache");

        ScheduleConfig conf = new ScheduleConfig();
        SWIGTYPE_p_MNN__Session session = interpreter.createSession(conf);

        TensorMap inputs = interpreter.getSessionInputAll(session);
        TensorMap outputs = interpreter.getSessionOutputAll(session);
        System.out.println("input count: " + inputs.size());
        System.out.println("output count: " + outputs.size());
        Tensor input = interpreter.getSessionInput(session, null);
        Tensor output = interpreter.getSessionOutput(session, null);
        System.out.println("input: " + input.batch() + "x" + input.channel() + "x" + input.height() + "x" + input.width());
        System.out.println("output: " + output.batch() + "x" + output.channel() + "x" + output.height() + "x" + output.width());

        interpreter.releaseModel();
    }

    public static void main(String[] args) {
        String model = "mnv2.mnn";
        if (args.length > 0) {
            model = args[0];
        }

        InterpreterTest test = new InterpreterTest();
        test.run(model);
    }
}

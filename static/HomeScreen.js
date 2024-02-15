function identify() {
    var raga = {notes: $("#notesid").val(),};
    $.ajax({
        url: "/raga",
        type: "POST",
        data: JSON.stringify(raga),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            //alert("Identification Success!!!! ");
            $("#resultid").text(data.raga);
        }

    });
};